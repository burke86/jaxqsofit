from __future__ import annotations

import os
import glob
from dataclasses import replace
from pathlib import Path

import extinction
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from jaxsedfit.filters import load_filter_curves

import jax
import jax.numpy as jnp
import optax
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import optax_to_numpyro

from .config import (
    ContinuumConfig,
    HostConfig,
    InferenceConfig,
    LineConfig,
    Observation,
    OutputConfig,
    PreprocessingConfig,
    PSFPhotometryData,
    FitConfig,
    PriorConfig,
    SpectroscopyData,
)
from .custom_components import (
    CustomComponentSpec,
    CustomLineComponentSpec,
    custom_component_site_names,
    custom_line_component_site_names,
    inject_default_custom_component_priors,
    inject_default_custom_line_component_priors,
    normalize_custom_components,
    normalize_custom_line_components,
)
from .defaults import build_default_bal_components, build_default_prior_config
from .model import (
    C_KMS,
    _continuum_output_waves_from_prior_config,
    _extract_line_table_from_prior_config,
    _format_wave_label,
    _get_sfd_query,
    _normalize_template_flux,
    _np_to_jnp,
    _spectrum_center_pivot,
    FSPSTemplateGrid,
    build_fsps_template_grid,
    build_tied_line_meta_from_linelist,
    qso_fsps_joint_model,
    reconstruct_posterior_components,
    unred,
)
from .results import FitResult, _PosteriorState, median_mapping

_SDSS_PSF_BANDS = ("u", "g", "r", "i", "z")
_SDSS_FILTER_CACHE = None


def _materialize_prior_config(prior_config) -> dict:
    """Return a mutable flat prior mapping for low-level model code."""
    if prior_config is None:
        return {}
    if isinstance(prior_config, PriorConfig):
        return prior_config.to_mapping()
    if hasattr(prior_config, "to_mapping"):
        return dict(prior_config.to_mapping())
    return dict(prior_config)


def _get_sdss_filters():
    """Load SDSS filter curves once and return a band->response mapping."""
    global _SDSS_FILTER_CACHE
    if _SDSS_FILTER_CACHE is None:
        filters = load_filter_curves([f"{band}_sdss" for band in _SDSS_PSF_BANDS])
        _SDSS_FILTER_CACHE = {band: filt for band, filt in zip(_SDSS_PSF_BANDS, filters)}
    return _SDSS_FILTER_CACHE


def _filter_wave_to_angstrom_array(value):
    """Return a filter wavelength grid as a float ndarray in Angstrom."""
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(u.AA), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _filter_wave_to_angstrom_scalar(value):
    """Return a scalar wavelength-like object as a float in Angstrom."""
    if hasattr(value, "to_value"):
        return float(value.to_value(u.AA))
    return float(value)


def _ab_mag_to_fnu(mag):
    """Convert AB magnitude to flux density in cgs units."""
    mag = np.asarray(mag, dtype=np.float64)
    return 10.0 ** (-0.4 * (mag + 48.60))


def _fnu_to_ab_mag(fnu):
    """Convert flux density in cgs units to AB magnitude."""
    fnu = np.asarray(fnu, dtype=np.float64)
    return -2.5 * np.log10(np.clip(fnu, 1e-300, None)) - 48.60


def _mw_band_attenuation_factor(wave_obs, filt_trans, ebv, r_v=3.1):
    """Return the AB-weighted Galactic attenuation factor through a filter."""
    wave_obs = np.asarray(wave_obs, dtype=np.float64)
    filt_trans = np.clip(np.asarray(filt_trans, dtype=np.float64), 0.0, None)
    if (not np.isfinite(ebv)) or ebv == 0.0:
        return 1.0

    a_lambda = extinction.fitzpatrick99(wave_obs, a_v=float(r_v) * float(ebv), r_v=float(r_v))
    attenuation = 10.0 ** (-0.4 * np.asarray(a_lambda, dtype=np.float64))
    inv_wave = 1.0 / np.clip(wave_obs, 1e-8, None)
    denom = float(np.trapezoid(filt_trans * inv_wave, wave_obs))
    if (not np.isfinite(denom)) or denom <= 0.0:
        return 1.0
    numer = float(np.trapezoid(filt_trans * attenuation * inv_wave, wave_obs))
    if (not np.isfinite(numer)) or numer <= 0.0:
        return 1.0
    return numer / denom

class JAXQSOFit:
    """Config-first spectral fitting interface for quasar spectra."""

    _POSTERIOR_BUNDLE_SUFFIX = ".h5"

    def __init__(self, config: "jaxqsofit.config.FitConfig"):
        """Initialize a config-first JAXQSOFit spectral fitter."""
        if not isinstance(config, FitConfig):
            raise TypeError("JAXQSOFit expects a FitConfig. Build one with jaxqsofit.FitConfig(...).")
        config.validate()
        self.config = config
        spec = config.spectroscopy
        obs = config.observation
        out = config.output
        psf = config.psf_photometry

        self.lam_in = np.asarray(spec.wave_obs, dtype=np.float64)
        self.flux_in = np.asarray(spec.fluxes, dtype=np.float64)
        if spec.errors is None:
            self.err_in = np.full_like(self.flux_in, 1e-6, dtype=np.float64)
        else:
            err_arr = np.asarray(spec.errors, dtype=np.float64)
            if err_arr.ndim == 0:
                self.err_in = np.full_like(self.flux_in, float(err_arr), dtype=np.float64)
            else:
                self.err_in = err_arr
        self.z = float(obs.redshift)
        self.wdisp = spec.wavelength_dispersion
        self.ra = -999 if obs.ra is None else float(obs.ra)
        self.dec = -999 if obs.dec is None else float(obs.dec)
        self.install_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = out.output_path
        self.filename = self._resolve_filename(filename=out.save_name or obs.object_id, ra=self.ra, dec=self.dec)
        self.psf_mags = None if psf is None else np.asarray(psf.magnitudes, dtype=np.float64)
        self.psf_mag_errs = None if psf is None else np.asarray(psf.magnitude_errors, dtype=np.float64)
        self.psf_mags_raw = None if psf is None else np.asarray(psf.magnitudes, dtype=np.float64)
        self.psf_mag_errs_raw = None if psf is None else np.asarray(psf.magnitude_errors, dtype=np.float64)
        self.psf_mags_dered = None
        self.psf_mag_errs_dered = None
        self.psf_bands = None if psf is None else list(psf.filter_names)
        if self.psf_bands is None and self.psf_mags is not None:
            self.psf_bands = ["u", "g", "r", "i", "z"][:len(self.psf_mags)]
        self.psf_filter_curves = None
        self.use_psf_phot = False
        self.ebv_mw = np.nan
        self._posterior_state = _PosteriorState()

    def _ensure_posterior_state(self) -> _PosteriorState:
        """Return the internal posterior state, creating it for legacy objects."""
        state = self.__dict__.get("_posterior_state")
        if state is None:
            state = _PosteriorState()
            self.__dict__["_posterior_state"] = state
        return state

    def _sync_posterior_state_from_legacy_attrs(self) -> None:
        """Fold legacy dict-loaded posterior attributes into ``_posterior_state``."""
        state = self._ensure_posterior_state()
        attr_map = {
            "numpyro_samples": "samples",
            "pred_out": "predictive",
            "pred_bands": "bands",
            "fig": "figure",
            "trace_fig": "trace_figure",
            "corner_fig": "corner_figure",
            "_loaded_posterior_path": "path",
            "_posterior_hydrated": "hydrated",
            "_resumed_from_samples": "resumed_from_samples",
        }
        for attr, field in attr_map.items():
            if attr in self.__dict__:
                value = self.__dict__.pop(attr)
                if field == "path" and value is not None:
                    value = Path(value)
                setattr(state, field, value)

    @property
    def numpyro_samples(self):
        """Posterior samples mirrored from the internal posterior state."""
        return self._ensure_posterior_state().samples

    @numpyro_samples.setter
    def numpyro_samples(self, value) -> None:
        self._ensure_posterior_state().samples = value

    @property
    def pred_out(self):
        """Posterior predictive outputs mirrored from the internal posterior state."""
        return self._ensure_posterior_state().predictive

    @pred_out.setter
    def pred_out(self, value) -> None:
        self._ensure_posterior_state().predictive = value

    @property
    def pred_bands(self):
        """Posterior uncertainty bands mirrored from the internal posterior state."""
        return self._ensure_posterior_state().bands

    @pred_bands.setter
    def pred_bands(self, value) -> None:
        self._ensure_posterior_state().bands = value

    @property
    def fig(self):
        """Main fitted-spectrum figure mirrored from the internal posterior state."""
        return self._ensure_posterior_state().figure

    @fig.setter
    def fig(self, value) -> None:
        self._ensure_posterior_state().figure = value

    @property
    def trace_fig(self):
        """Trace figure mirrored from the internal posterior state."""
        return self._ensure_posterior_state().trace_figure

    @trace_fig.setter
    def trace_fig(self, value) -> None:
        self._ensure_posterior_state().trace_figure = value

    @property
    def corner_fig(self):
        """Corner figure mirrored from the internal posterior state."""
        return self._ensure_posterior_state().corner_figure

    @corner_fig.setter
    def corner_fig(self, value) -> None:
        self._ensure_posterior_state().corner_figure = value

    @property
    def _loaded_posterior_path(self):
        """Loaded posterior bundle path mirrored from the internal posterior state."""
        return self._ensure_posterior_state().path

    @_loaded_posterior_path.setter
    def _loaded_posterior_path(self, value) -> None:
        self._ensure_posterior_state().path = None if value is None else Path(value)

    @property
    def _posterior_hydrated(self) -> bool:
        """Whether posterior-derived products have been reconstructed."""
        return bool(self._ensure_posterior_state().hydrated)

    @_posterior_hydrated.setter
    def _posterior_hydrated(self, value: bool) -> None:
        self._ensure_posterior_state().hydrated = bool(value)

    @property
    def _resumed_from_samples(self) -> bool:
        """Whether this fitter was loaded from a posterior bundle."""
        return bool(self._ensure_posterior_state().resumed_from_samples)

    @_resumed_from_samples.setter
    def _resumed_from_samples(self, value: bool) -> None:
        self._ensure_posterior_state().resumed_from_samples = bool(value)

    @classmethod
    def from_arrays(
        cls,
        *,
        lam,
        flux,
        err=None,
        z=0.0,
        ra=None,
        dec=None,
        filename=None,
        output_path=None,
        wdisp=None,
        psf_mags=None,
        psf_mag_errs=None,
        psf_bands=None,
    ):
        """Build a config-first fitter from raw arrays."""
        psf = None
        if psf_mags is not None and psf_mag_errs is not None:
            psf = PSFPhotometryData(
                magnitudes=psf_mags,
                magnitude_errors=psf_mag_errs,
                filter_names=tuple(psf_bands) if psf_bands is not None else ("u", "g", "r", "i", "z")[:len(psf_mags)],
            )
        cfg = FitConfig(
            observation=Observation(
                object_id=cls._resolve_filename(filename=filename, ra=-999 if ra is None else ra, dec=-999 if dec is None else dec),
                redshift=float(z),
                ra=None if ra in (None, -999) else float(ra),
                dec=None if dec in (None, -999) else float(dec),
            ),
            spectroscopy=SpectroscopyData(
                wave_obs=lam,
                fluxes=flux,
                errors=err,
                wavelength_dispersion=wdisp,
            ),
            psf_photometry=psf,
            output=OutputConfig(output_path=output_path, save_name=filename),
        )
        return cls(cfg)

    @staticmethod
    def _resolve_filename(filename=None, ra=-999, dec=-999):
        """Resolve a filesystem-safe basename for outputs."""
        if filename is not None and str(filename).strip() != "":
            return str(filename).strip()
        try:
            ra_f = float(ra)
            dec_f = float(dec)
        except Exception:
            return "result"
        if np.isfinite(ra_f) and np.isfinite(dec_f) and (ra_f != -999) and (dec_f != -999):
            return f"ra{ra_f:.5f}_dec{dec_f:.5f}"
        return "result"

    def _predictive_return_sites(self, custom_components=None, custom_line_components=None):
        """Return posterior predictive sites needed for summaries and plots."""
        return_sites = [
            'f_pl_model',
            'f_fe_mgii_model',
            'f_fe_balmer_model',
            'f_bc_model',
            'f_poly_model',
            'reddening_a2500',
            'agn_model',
            'gal_model',
            'line_model_broad',
            'line_model_narrow',
            'line_component_profiles',
            'line_model',
            'continuum_model',
            'model',
            'fsps_weights',
            'line_amp_per_component',
            'line_mu_per_component',
            'line_sig_per_component',
            'delta_m_psf',
            'eta_psf',
            'scale_psf',
            'agn_model_psf',
            'gal_model_psf',
            'line_model_broad_psf',
            'line_model_narrow_psf',
            'line_component_profiles_psf',
            'line_model_psf',
            'psf_model',
        ]
        for wave_lum in _continuum_output_waves_from_prior_config(
            getattr(self, "_fit_prior_config", None)
        ):
            wave_label = _format_wave_label(wave_lum)
            return_sites.append(f"log_lambda_Llambda_{wave_label}_agn")
        return_sites += custom_component_site_names(custom_components)
        return_sites += custom_line_component_site_names(custom_line_components)
        return return_sites

    def _prepare_psf_photometry(
        self,
        wave_obs,
        psf_mags=None,
        psf_mag_errs=None,
        psf_bands=None,
        use_psf_phot=False,
        min_filter_coverage=0.97,
    ):
        """Validate PSF photometry and project filters onto the spectral grid.

        JAXQSOFit only fits the spectrum. PSF photometry is therefore a
        spectral-recalibration constraint, not a general SED likelihood. Bands
        with no transmission overlap on the observed spectral wavelength grid
        are dropped; use ``jaxsedfit`` for full joint spectrum + broadband SED
        modeling.
        """
        if psf_mags is not None:
            self.psf_mags = np.asarray(psf_mags, dtype=np.float64)
            self.psf_mags_raw = np.asarray(psf_mags, dtype=np.float64)
        if psf_mag_errs is not None:
            self.psf_mag_errs = np.asarray(psf_mag_errs, dtype=np.float64)
            self.psf_mag_errs_raw = np.asarray(psf_mag_errs, dtype=np.float64)
        if psf_bands is not None:
            self.psf_bands = list(psf_bands)
        if self.psf_bands is None and self.psf_mags is not None:
            self.psf_bands = list(_SDSS_PSF_BANDS[:len(self.psf_mags)])

        if (not use_psf_phot) or self.psf_mags is None or self.psf_mag_errs is None:
            self.use_psf_phot = False
            self.psf_filter_curves = None
            self.psf_mags_dered = None
            self.psf_mag_errs_dered = None
            return None, None, None, None, False

        mags = np.asarray(self.psf_mags, dtype=np.float64)
        errs = np.asarray(self.psf_mag_errs, dtype=np.float64)
        bands = list(self.psf_bands) if self.psf_bands is not None else list(_SDSS_PSF_BANDS[:len(mags)])
        if len(mags) != len(errs) or len(mags) != len(bands):
            raise ValueError("psf_mags, psf_mag_errs, and psf_bands must have the same length.")

        valid = np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        filters = _get_sdss_filters()

        keep_mags = []
        keep_errs = []
        keep_bands = []
        keep_trans = []
        keep_coverage = []
        for band, mag, err, is_valid in zip(bands, mags, errs, valid):
            if not is_valid:
                continue
            if band not in filters:
                raise ValueError(f"Unsupported PSF photometry band '{band}'. Supported bands: {_SDSS_PSF_BANDS}.")

            filt = filters[band]
            filt_wave = _filter_wave_to_angstrom_array(filt.wave)
            filt_trans = np.asarray(filt.transmission, dtype=np.float64)
            trans_on_wave = np.interp(wave_obs, filt_wave, filt_trans, left=0.0, right=0.0)

            full_norm = float(np.trapezoid(np.clip(filt_trans, 0.0, None), filt_wave))
            covered_norm = float(np.trapezoid(np.clip(trans_on_wave, 0.0, None), wave_obs))
            coverage = (covered_norm / full_norm) if full_norm > 0 else 0.0
            if coverage < float(min_filter_coverage):
                continue

            keep_mags.append(float(mag))
            keep_errs.append(float(err))
            keep_bands.append(str(band))
            keep_trans.append(np.asarray(trans_on_wave, dtype=np.float64))
            keep_coverage.append(float(coverage))

        if len(keep_bands) == 0:
            self.use_psf_phot = False
            self.psf_filter_curves = None
            self.psf_mags = None
            self.psf_mag_errs = None
            self.psf_mags_raw = None
            self.psf_mag_errs_raw = None
            self.psf_mags_dered = None
            self.psf_mag_errs_dered = None
            self.psf_bands = None
            return None, None, None, None, False

        raw_mags = np.asarray(keep_mags, dtype=np.float64)
        raw_errs = np.asarray(keep_errs, dtype=np.float64)
        dered_mags = raw_mags.copy()
        apply_dered = bool(getattr(self, "_fit_deredden", False)) and np.isfinite(getattr(self, "ebv_mw", np.nan))
        if apply_dered and float(self.ebv_mw) != 0.0:
            band_atten = np.asarray(
                [_mw_band_attenuation_factor(wave_obs, trans, self.ebv_mw) for trans in keep_trans],
                dtype=np.float64,
            )
            fnu_obs = _ab_mag_to_fnu(raw_mags)
            fnu_dered = fnu_obs / np.clip(band_atten, 1e-30, None)
            dered_mags = _fnu_to_ab_mag(fnu_dered)

        self.psf_mags_raw = raw_mags
        self.psf_mag_errs_raw = raw_errs
        self.psf_mags_dered = dered_mags
        self.psf_mag_errs_dered = raw_errs.copy()
        self.psf_mags = dered_mags
        self.psf_mag_errs = raw_errs
        self.psf_bands = keep_bands
        self.psf_filter_curves = {
            "bands": tuple(keep_bands),
            "trans": np.asarray(keep_trans, dtype=np.float64),
            "coverage": np.asarray(keep_coverage, dtype=np.float64),
        }
        self.use_psf_phot = True
        return (
            self.psf_mags,
            self.psf_mag_errs,
            self.psf_bands,
            {"trans": self.psf_filter_curves["trans"]},
            True,
        )

    def _posterior_bundle_path(self, save_name=None, save_path=None):
        """Return the compressed on-disk path for a saved posterior bundle."""
        out_name = self._normalize_posterior_bundle_name(
            f"{self.filename}_samples" if save_name is None else save_name
        )
        out_dir = self.output_path if save_path is None else save_path
        if out_dir is None:
            out_dir = '.'
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, out_name)

    def _intrinsic_powerlaw_draws(self, wave_out=None, apply_psf_scale=False):
        """Return posterior draws for the intrinsic AGN power law on ``wave_out``."""
        samples = getattr(self, 'numpyro_samples', None)
        if samples is None or 'PL_slope' not in samples:
            return None

        wave_eval = np.asarray(self.wave if wave_out is None else wave_out, dtype=float)
        if wave_eval.ndim != 1 or wave_eval.size == 0 or not np.all(np.isfinite(wave_eval)):
            return None

        pl_norm = np.asarray(samples['PL_norm'], dtype=float).reshape(-1)
        pl_slope = np.asarray(samples['PL_slope'], dtype=float).reshape(-1)
        if pl_norm.size == 0 or pl_slope.size == 0:
            return None

        n = min(pl_norm.size, pl_slope.size)
        if n == 0:
            return None
        pl_norm = pl_norm[:n]
        pl_slope = pl_slope[:n]

        prior_config = getattr(self, '_fit_prior_config', None) or {}
        pivot = prior_config.get('PL_pivot', None)
        if pivot is None:
            pivot = 0.5 * (wave_eval[0] + wave_eval[-1])
        pivot = max(float(pivot), 1e-8)

        x = np.clip(wave_eval / pivot, 1e-8, None)
        draws = pl_norm[:, None] * (x[None, :] ** pl_slope[:, None])
        if apply_psf_scale:
            psf_scale = float(getattr(self, 'scale_psf', np.nan))
            if np.isfinite(psf_scale):
                draws = psf_scale * draws
        return draws
    @classmethod
    def _normalize_posterior_bundle_name(cls, name):
        """Normalize posterior bundle names to the enforced ``.h5`` suffix."""
        name = str(name)
        if name.endswith(cls._POSTERIOR_BUNDLE_SUFFIX):
            return name
        return name + cls._POSTERIOR_BUNDLE_SUFFIX

    @staticmethod
    def _bundle_excluded_keys():
        """Return object attributes intentionally omitted from saved bundles."""
        return {
            "numpyro_mcmc",
            "svi",
            "svi_state",
            "fig",
            "trace_fig",
            "corner_fig",
            "fsps_grid",
            "fe_uv",
            "fe_op",
            "pred_out",
        }

    @staticmethod
    def _is_matplotlib_state(value):
        """Return True when value is a matplotlib figure/axes object."""
        classes = []
        fig_cls = getattr(getattr(matplotlib, "figure", None), "Figure", None)
        axes_cls = getattr(getattr(matplotlib, "axes", None), "Axes", None)
        if isinstance(fig_cls, type):
            classes.append(fig_cls)
        if isinstance(axes_cls, type):
            classes.append(axes_cls)
        if len(classes) == 0:
            return False
        return isinstance(value, tuple(classes))

    @classmethod
    def _exclude_from_posterior_bundle(cls, key, value):
        """Return True when an attribute should be skipped during bundle save."""
        if key in cls._bundle_excluded_keys():
            return True
        if key.startswith("_pred_"):
            return True
        if cls._is_matplotlib_state(value):
            return True
        return False

    @staticmethod
    def _serialize_for_hdf5(value):
        """Recursively convert model state into HDF5-serializable objects."""
        if isinstance(value, CustomComponentSpec):
            return value.to_state()
        if isinstance(value, CustomLineComponentSpec):
            return value.to_state()
        if hasattr(value, "to_mapping"):
            return JAXQSOFit._serialize_for_hdf5(value.to_mapping())
        if isinstance(value, dict):
            return {str(k): JAXQSOFit._serialize_for_hdf5(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(JAXQSOFit._serialize_for_hdf5(v) for v in value)
        if isinstance(value, np.ndarray) and value.dtype == object:
            return {
                "__ndarray_object__": True,
                "shape": tuple(int(x) for x in value.shape),
                "items": [JAXQSOFit._serialize_for_hdf5(v) for v in value.ravel(order="C").tolist()],
            }
        if isinstance(value, (np.ndarray, np.generic)):
            return np.asarray(value)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return np.asarray(value)
        return value

    @staticmethod
    def _deserialize_from_hdf5(value):
        """Rebuild custom serialized objects after reading HDF5 state."""
        if isinstance(value, dict):
            if value.get("__custom_component__", False):
                return CustomComponentSpec.from_state(value)
            if value.get("__custom_line_component__", False):
                return CustomLineComponentSpec.from_state(value)
            if value.get("__ndarray_object__", False):
                items = [JAXQSOFit._deserialize_from_hdf5(v) for v in value["items"]]
                arr = np.asarray(items, dtype=object)
                return arr.reshape(tuple(value["shape"]))
            return {k: JAXQSOFit._deserialize_from_hdf5(v) for k, v in value.items()}
        if isinstance(value, list):
            return [JAXQSOFit._deserialize_from_hdf5(v) for v in value]
        if isinstance(value, tuple):
            return tuple(JAXQSOFit._deserialize_from_hdf5(v) for v in value)
        return value

    @staticmethod
    def _hdf5_scalar_string_dtype():
        """Return the UTF-8 scalar string dtype used in HDF5 bundles."""
        return h5py.string_dtype(encoding="utf-8")

    @classmethod
    def _write_hdf5_node(cls, parent, name, value):
        """Write one recursively serialized Python value into an HDF5 group."""
        value = cls._serialize_for_hdf5(value)
        if value is None:
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "none"
            return

        if isinstance(value, dict):
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "dict"
            for idx, (k, v) in enumerate(value.items()):
                item_grp = grp.create_group(f"item_{idx:08d}")
                cls._write_hdf5_node(item_grp, "key", str(k))
                cls._write_hdf5_node(item_grp, "value", v)
            return

        if isinstance(value, list):
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "list"
            for idx, item in enumerate(value):
                cls._write_hdf5_node(grp, f"item_{idx:08d}", item)
            return

        if isinstance(value, tuple):
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "tuple"
            for idx, item in enumerate(value):
                cls._write_hdf5_node(grp, f"item_{idx:08d}", item)
            return

        if isinstance(value, np.ndarray):
            ds_kwargs = {}
            if value.ndim > 0:
                ds_kwargs["compression"] = "gzip"
                ds_kwargs["shuffle"] = True
            ds = parent.create_dataset(name, data=value, **ds_kwargs)
            ds.attrs["node_type"] = "ndarray"
            return

        if isinstance(value, bool):
            ds = parent.create_dataset(name, data=np.bool_(value))
            ds.attrs["node_type"] = "scalar_bool"
            return

        if isinstance(value, int):
            ds = parent.create_dataset(name, data=np.int64(value))
            ds.attrs["node_type"] = "scalar_int"
            return

        if isinstance(value, float):
            ds = parent.create_dataset(name, data=np.float64(value))
            ds.attrs["node_type"] = "scalar_float"
            return

        if isinstance(value, str):
            ds = parent.create_dataset(name, data=np.array(value, dtype=cls._hdf5_scalar_string_dtype()))
            ds.attrs["node_type"] = "scalar_str"
            return

        raise TypeError(f"Unsupported value type in posterior bundle: {type(value)!r}")

    @classmethod
    def _read_hdf5_node(cls, parent, name):
        """Read one recursively serialized Python value from an HDF5 group."""
        node = parent[name]
        if isinstance(node, h5py.Dataset):
            node_type = node.attrs.get("node_type", "ndarray")
            if isinstance(node_type, bytes):
                node_type = node_type.decode("utf-8")
            if node_type == "scalar_str":
                return node.asstr()[()]
            value = node[()]
            if node_type == "scalar_bool":
                return bool(value)
            if node_type == "scalar_int":
                return int(value)
            if node_type == "scalar_float":
                return float(value)
            return np.asarray(value)

        node_type = node.attrs.get("node_type", "")
        if isinstance(node_type, bytes):
            node_type = node_type.decode("utf-8")
        if node_type == "none":
            return None
        if node_type == "dict":
            out = {}
            for item_name in sorted(node.keys()):
                item_grp = node[item_name]
                key = cls._read_hdf5_node(item_grp, "key")
                out[str(key)] = cls._read_hdf5_node(item_grp, "value")
            return out
        if node_type == "list":
            return [cls._read_hdf5_node(node, item_name) for item_name in sorted(node.keys())]
        if node_type == "tuple":
            return tuple(cls._read_hdf5_node(node, item_name) for item_name in sorted(node.keys()))
        raise TypeError(f"Unsupported HDF5 node type in posterior bundle: {node_type!r}")

    @staticmethod
    def _sample_bundle_meta_keys():
        """Return metadata keys persisted in sample-only bundles."""
        return {
            "lam_in",
            "flux_in",
            "err_in",
            "z",
            "ra",
            "dec",
            "filename",
            "output_path",
            "wdisp",
            "wave",
            "flux",
            "err",
            "wave_prereduced",
            "flux_prereduced",
            "fe_uv_wave",
            "fe_uv_flux",
            "fe_op_wave",
            "fe_op_flux",
            "psf_mags",
            "psf_mag_errs",
            "psf_mags_raw",
            "psf_mag_errs_raw",
            "psf_mags_dered",
            "psf_mag_errs_dered",
            "psf_bands",
            "psf_filter_curves",
            "use_psf_phot",
            "verbose",
            "save_fig",
            "_fit_deredden",
            "_fit_decompose_host",
            "_fit_fit_lines",
            "_fit_fit_pl",
            "_fit_fit_fe",
            "_fit_fit_bc",
            "_fit_fit_bal",
            "_fit_fit_poly",
            "_fit_fit_reddening",
            "_fit_fit_poly_order",
            "_fit_mask_lya_forest",
            "_fit_inference_method",
            "_fit_fsps_age_grid",
            "_fit_fsps_logzsol_grid",
            "_fit_fsps_template_norms",
            "_fit_prior_config",
            "_fit_dsps_ssp_fn",
            "_fit_use_psf_phot",
            "_fit_custom_components",
            "_fit_custom_line_components",
        }

    def _collect_sample_bundle_meta(self):
        """Collect minimal metadata for sample-only bundle persistence."""
        if not hasattr(self, "numpyro_samples") or self.numpyro_samples is None:
            raise RuntimeError("No posterior samples available. Run fit() before saving a posterior bundle.")
        keys = self._sample_bundle_meta_keys()
        meta = {}
        for key in keys:
            if key not in self.__dict__:
                continue
            value = self.__dict__[key]
            if self._exclude_from_posterior_bundle(key, value):
                continue
            meta[key] = self._serialize_for_hdf5(value)
        return meta

    @staticmethod
    def _empty_tied_line_meta():
        """Return an empty tied-line metadata payload."""
        return {
            'n_lines': 0,
            'n_vgroups': 0,
            'n_wgroups': 0,
            'n_fgroups': 0,
            'ln_lambda0': _np_to_jnp(np.array([], dtype=float)),
            'vgroup': np.array([], dtype=int),
            'wgroup': np.array([], dtype=int),
            'fgroup': np.array([], dtype=int),
            'flux_ratio': np.array([], dtype=float),
            'dmu_init_group': np.array([], dtype=float),
            'dmu_min_group': np.array([], dtype=float),
            'dmu_max_group': np.array([], dtype=float),
            'sig_init_group': np.array([], dtype=float),
            'sig_min_group': np.array([], dtype=float),
            'sig_max_group': np.array([], dtype=float),
            'amp_init_group': np.array([], dtype=float),
            'amp_min_group': np.array([], dtype=float),
            'amp_max_group': np.array([], dtype=float),
            'names': [],
            'compnames': [],
            'line_lambda': np.array([], dtype=float),
        }

    @staticmethod
    def _require_posterior_bundle_fsps_metadata(state):
        """Return required FSPS bundle metadata or raise on incomplete bundles."""
        required_keys = (
            "_fit_fsps_age_grid",
            "_fit_fsps_logzsol_grid",
            "_fit_dsps_ssp_fn",
        )
        missing = [key for key in required_keys if key not in state or state[key] is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "Posterior bundle is missing required FSPS metadata for hydration: "
                f"{joined}."
            )

        age_grid_gyr = tuple(np.asarray(state["_fit_fsps_age_grid"], dtype=float).tolist())
        logzsol_grid = tuple(np.asarray(state["_fit_fsps_logzsol_grid"], dtype=float).tolist())
        dsps_ssp_fn = state["_fit_dsps_ssp_fn"]
        if len(age_grid_gyr) == 0 or len(logzsol_grid) == 0:
            raise ValueError("Posterior bundle FSPS metadata must define non-empty age and metallicity grids.")
        if not isinstance(dsps_ssp_fn, str) or len(dsps_ssp_fn) == 0:
            raise ValueError("Posterior bundle FSPS metadata must include a non-empty dsps_ssp_fn.")
        return age_grid_gyr, logzsol_grid, dsps_ssp_fn

    @staticmethod
    def _validate_fsps_weights_shape(pred_out, expected_templates, context):
        """Ensure hydrated or reconstructed FSPS weights match the expected basis width."""
        if pred_out is None or "fsps_weights" not in pred_out:
            raise ValueError(f"{context} requires pred_out['fsps_weights'] to be present.")
        fsps_weights = np.asarray(pred_out["fsps_weights"], dtype=float)
        if fsps_weights.ndim != 2:
            raise ValueError(
                f"{context} requires pred_out['fsps_weights'] to be a 2D array; "
                f"got shape {fsps_weights.shape}."
            )
        if fsps_weights.shape[1] != int(expected_templates):
            raise ValueError(
                f"{context} requires pred_out['fsps_weights'] width {expected_templates}, "
                f"got {fsps_weights.shape[1]}."
            )
        return fsps_weights

    def _ensure_hydrated_from_samples(self):
        """Rebuild posterior-derived component products from saved samples."""
        if bool(getattr(self, "_posterior_hydrated", False)):
            return
        has_cached = (
            hasattr(self, "model_total")
            and hasattr(self, "f_conti_model")
            and hasattr(self, "f_line_model")
            and hasattr(self, "host")
            and self.pred_bands is not None
        )
        if has_cached:
            self._posterior_hydrated = True
            return
        if not hasattr(self, "numpyro_samples") or self.numpyro_samples is None:
            raise RuntimeError("No posterior samples available for hydration.")
        if not hasattr(self, "wave") or not hasattr(self, "flux") or not hasattr(self, "err"):
            raise RuntimeError("Missing fitted spectrum context (wave/flux/err) for hydration.")

        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)
        if wave.ndim != 1 or wave.size < 2:
            raise RuntimeError("Invalid fitted wavelength grid for hydration.")

        prior_config = getattr(self, "_fit_prior_config", None)
        if prior_config is None:
            prior_config = _materialize_prior_config(build_default_prior_config(flux))
        custom_components = normalize_custom_components(getattr(self, "_fit_custom_components", ()))
        custom_line_components = normalize_custom_line_components(getattr(self, "_fit_custom_line_components", ()))
        prior_config = inject_default_custom_component_priors(prior_config, flux, custom_components)
        prior_config = inject_default_custom_line_component_priors(prior_config, flux, custom_line_components)
        conti_priors = prior_config.get("conti_priors", {})

        use_lines = bool(getattr(self, "_fit_fit_lines", True))
        line_table = _extract_line_table_from_prior_config(prior_config)
        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)
        else:
            tied_line_meta = self._empty_tied_line_meta()
        if use_lines and line_table is None and len(custom_line_components) == 0:
            raise RuntimeError("Hydration requires line priors/table when fit_lines=True.")

        age_grid_gyr, logzsol_grid, dsps_ssp_fn = self._require_posterior_bundle_fsps_metadata(self.__dict__)
        decompose_host = bool(getattr(self, "_fit_decompose_host", True))
        fsps_grid = self._build_fsps_grid_for_fit(
            wave=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
            decompose_host=decompose_host,
            z_qso=float(getattr(self, "z", 0.0)),
        )
        self.tied_line_meta = tied_line_meta

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples={k: jnp.asarray(v) for k, v in self.numpyro_samples.items()},
            return_sites=self._predictive_return_sites(
                custom_components=custom_components,
                custom_line_components=custom_line_components,
            ),
        )
        rng_key = jax.random.PRNGKey(0)
        pred_out = pred(
            rng_key,
            wave=wave,
            flux=None,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=bool(getattr(self, "_fit_fit_pl", True)),
            fit_fe=bool(getattr(self, "_fit_fit_fe", True)),
            fit_bc=bool(getattr(self, "_fit_fit_bc", True)),
            fit_poly=bool(getattr(self, "_fit_fit_poly", False)),
            fit_reddening=bool(getattr(self, "_fit_fit_reddening", False)),
            fit_poly_order=int(getattr(self, "_fit_fit_poly_order", 2)),
            z_qso=float(getattr(self, "z", 0.0)),
            psf_mags=getattr(self, "psf_mags", None),
            psf_mag_errs=getattr(self, "psf_mag_errs", None),
            psf_filter_curves=getattr(self, "psf_filter_curves", None),
            use_psf_phot=bool(getattr(self, "_fit_use_psf_phot", getattr(self, "use_psf_phot", False))),
            custom_components=custom_components,
            custom_line_components=custom_line_components,
        )
        self._validate_fsps_weights_shape(
            pred_out,
            expected_templates=fsps_grid.templates.shape[1],
            context="Hydrated posterior state",
        )
        self._consume_posterior_outputs(
            samples=self.numpyro_samples,
            pred_out=pred_out,
            fsps_grid=fsps_grid,
            tied_line_meta=tied_line_meta,
            use_lines=use_lines,
            decompose_host=decompose_host,
        )

    def save_posterior_bundle(self, save_name=None, save_path=None, *, _state: _PosteriorState | None = None):
        """Persist posterior samples plus minimal metadata for compact reloads."""
        state = self._ensure_posterior_state() if _state is None else _state
        if state.samples is None:
            raise RuntimeError("No posterior samples available. Run fit() before saving a posterior bundle.")
        meta = self._collect_sample_bundle_meta()

        out_file = self._posterior_bundle_path(save_name=save_name, save_path=save_path)
        with h5py.File(out_file, "w") as h5f:
            h5f.attrs["posterior_bundle_format"] = "jaxqsofit_samples_meta_v1"
            samples_grp = h5f.create_group("samples")
            for name, draws in state.samples.items():
                arr = np.asarray(draws)
                ds_kwargs = {}
                if arr.ndim > 0:
                    ds_kwargs["compression"] = "gzip"
                    ds_kwargs["shuffle"] = True
                samples_grp.create_dataset(str(name), data=arr, **ds_kwargs)
            meta_grp = h5f.create_group("meta")
            for key, value in meta.items():
                self._write_hdf5_node(meta_grp, str(key), value)
        print(f"Saved posterior bundle: {out_file}")
        state.path = Path(out_file)
        return out_file

    def save(self, path=None, *, save_name=None, _state: _PosteriorState | None = None):
        """Persist posterior samples and fit metadata to a compact bundle."""
        return self.save_posterior_bundle(save_name=save_name, save_path=path, _state=_state)

    @staticmethod
    def _build_fsps_grid_for_fit(wave, age_grid_gyr, logzsol_grid, dsps_ssp_fn, decompose_host, z_qso=0.0):
        """Build the host-template grid only when host decomposition is enabled."""
        if decompose_host:
            return build_fsps_template_grid(
                wave_out=wave,
                age_grid_gyr=age_grid_gyr,
                logzsol_grid=logzsol_grid,
                dsps_ssp_fn=dsps_ssp_fn,
                z_qso=z_qso,
            )

        class _DummyFSPSGrid:
            """Minimal FSPS-like grid used when host decomposition is disabled."""
            pass

        wave = np.asarray(wave, dtype=float)
        age_grid_gyr = np.asarray(age_grid_gyr, dtype=float)
        logzsol_grid = np.asarray(logzsol_grid, dtype=float)
        grid = _DummyFSPSGrid()
        grid.wave = wave
        n_templates = int(len(age_grid_gyr) * len(logzsol_grid))
        grid.templates = np.zeros((len(wave), n_templates), dtype=float)
        grid.template_meta = []
        for logz in logzsol_grid:
            for age in age_grid_gyr:
                grid.template_meta.append({
                    'tage_gyr': float(age),
                    'logzsol': float(logz),
                    'norm': 1.0,
                    'dsps_lgmet': np.nan,
                    'dsps_lg_age_gyr': np.nan,
                })
        grid.age_grid_gyr = age_grid_gyr
        grid.logzsol_grid = logzsol_grid
        grid.host_basis_jax = None
        grid.t_obs_gyr = None
        return grid

    @classmethod
    def load_from_samples(
        cls,
        filename=None,
        output_path=None,
        save_name=None,
        plot_fig=True,
        plot_diagnostics=True,
        kwargs_plot=None,
        diagnostics_kwargs=None,
    ):
        """Load a compressed HDF5 posterior bundle and return a JAXQSOFit object."""
        if save_name is not None:
            bundle_name = cls._normalize_posterior_bundle_name(save_name)
            bundle_dir = '.' if output_path is None else output_path
            bundle_path = os.path.join(bundle_dir, bundle_name)
            resolved_name = cls._resolve_filename(filename=filename)
        elif filename is not None:
            resolved_name = cls._resolve_filename(filename=filename)
            bundle_name = cls._normalize_posterior_bundle_name(f"{resolved_name}_samples")
            bundle_dir = '.' if output_path is None else output_path
            bundle_path = os.path.join(bundle_dir, bundle_name)
        else:
            bundle_dir = '.' if output_path is None else output_path
            matches = sorted(glob.glob(os.path.join(bundle_dir, f"*_samples{cls._POSTERIOR_BUNDLE_SUFFIX}")))
            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No compressed posterior bundle (*.h5) found under: {bundle_dir}. "
                    "Pass filename=..., output_path=..., or save_name=... explicitly."
                )
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple compressed posterior bundles (*.h5) found under: {bundle_dir}. "
                    "Pass filename=... or save_name=... explicitly."
                )
            bundle_path = matches[0]
            bundle_name = os.path.basename(bundle_path)
            suffix = f"_samples{cls._POSTERIOR_BUNDLE_SUFFIX}"
            resolved_name = bundle_name[: -len(suffix)] if bundle_name.endswith(suffix) else bundle_name

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Posterior bundle not found: {bundle_path}")

        with h5py.File(bundle_path, "r") as h5f:
            if "samples" in h5f and "meta" in h5f:
                samples = {k: np.asarray(h5f["samples"][k][()]) for k in h5f["samples"].keys()}
                meta = {k: cls._read_hdf5_node(h5f["meta"], k) for k in h5f["meta"].keys()}
                meta = cls._deserialize_from_hdf5(meta)
                cls._require_posterior_bundle_fsps_metadata(meta)
                state = dict(meta)
                state["numpyro_samples"] = samples
                state["_posterior_hydrated"] = False
            elif "state" in h5f:
                # Backward-compatible read for older .h5 bundles.
                state = cls._read_hdf5_node(h5f, "state")
                state = cls._deserialize_from_hdf5(state)
            else:
                raise ValueError(f"Unsupported posterior bundle schema: {bundle_path}")

        obj = cls.from_arrays(
            lam=state["lam_in"],
            flux=state["flux_in"],
            err=state.get("err_in"),
            z=state.get("z", 0.0),
            ra=state.get("ra", -999),
            dec=state.get("dec", -999),
            filename=state.get("filename", resolved_name),
            output_path=output_path if output_path is not None else state.get("output_path"),
            wdisp=state.get("wdisp"),
            psf_mags=state.get("psf_mags_raw", state.get("psf_mags")),
            psf_mag_errs=state.get("psf_mag_errs_raw", state.get("psf_mag_errs")),
            psf_bands=state.get("psf_bands"),
        )
        obj.__dict__.update(state)
        obj._sync_posterior_state_from_legacy_attrs()
        obj._resumed_from_samples = True
        obj.install_path = os.path.dirname(os.path.abspath(__file__))
        if not hasattr(obj, "verbose"):
            obj.verbose = False
        if not hasattr(obj, "save_fig"):
            obj.save_fig = False
        if not hasattr(obj, "SN_ratio_conti"):
            obj.SN_ratio_conti = np.nan
        obj._loaded_posterior_path = bundle_path
        obj._ensure_hydrated_from_samples()

        if plot_fig:
            plot_kwargs = {} if kwargs_plot is None else dict(kwargs_plot)
            if "show_plot" not in plot_kwargs:
                plot_kwargs["show_plot"] = False
            obj.plot_fig(**plot_kwargs)
        if plot_diagnostics:
            diag_kwargs = {} if diagnostics_kwargs is None else dict(diagnostics_kwargs)
            obj.plot_mcmc_diagnostics(**diag_kwargs)
        return obj

    load = load_from_samples

    def _make_result(
        self,
        *,
        method: str | None = None,
        path=None,
        figure=None,
    ) -> FitResult:
        """Build a public result object from the current mirrored fit state."""
        state = self._ensure_posterior_state()
        if method is not None:
            state.method = str(method)
        if path is not None:
            state.path = Path(path)
        if figure is not None:
            state.figure = figure
        samples = state.samples
        median = median_mapping(samples)
        return FitResult(
            fitter=self,
            samples=samples,
            median=median,
            method=str(state.method if state.method is not None else getattr(self, "_fit_inference_method", "unknown")),
            summary=dict(median),
            path=state.path,
            figure=state.figure,
            _state=state,
        )

    @classmethod
    def load_result(cls, *args, **kwargs) -> FitResult:
        """Load a posterior bundle and wrap it in a :class:`FitResult`."""
        fitter = cls.load(*args, **kwargs)
        return fitter._make_result(
            method=getattr(fitter, "_fit_inference_method", "loaded"),
            path=getattr(fitter, "_loaded_posterior_path", None),
        )

    def fit(self, *, verbose=True, kwargs_plot=None):
        """Run preprocessing, inference, persistence, and plotting.

        The public API is configuration-first: construct ``JAXQSOFit`` with a
        :class:`jaxqsofit.config.FitConfig`, then call ``fit()``. Model choices,
        preprocessing, inference settings, output behavior, PSF recalibration
        data, and priors all live on the config object.

        Parameters
        ----------
        verbose : bool, optional
            Verbose optimizer output where applicable.
        kwargs_plot : dict or None, optional
            Extra keyword arguments passed to :meth:`plot_fig`.

        Returns
        -------
        FitResult
            Result object exposing samples, medians, persistence, and plotting
            helpers while the fitter keeps mirrored posterior state.
        """

        cfg = self.config
        obs_cfg = cfg.observation
        prep_cfg = cfg.preprocessing
        cont_cfg = cfg.continuum
        host_cfg = cfg.host
        line_cfg = cfg.lines
        infer_cfg = cfg.inference
        out_cfg = cfg.output
        psf_cfg = cfg.psf_photometry

        name = out_cfg.save_name
        deredden = bool(obs_cfg.apply_mw_deredden)
        wave_range = prep_cfg.wave_range
        wave_mask = prep_cfg.wave_mask
        mask_lya_forest = bool(prep_cfg.mask_lya_forest)
        fit_lines = bool(line_cfg.enabled)
        decompose_host = bool(host_cfg.enabled)
        fit_pl = bool(cont_cfg.fit_power_law)
        fit_fe = bool(cont_cfg.fit_feii)
        fit_bc = bool(cont_cfg.fit_balmer_continuum)
        fit_bal = bool(cont_cfg.fit_bal_absorption)
        fit_poly = bool(cont_cfg.fit_polynomial_tilt)
        fit_reddening = bool(cont_cfg.fit_reddening)
        fit_poly_order = int(cont_cfg.polynomial_order)
        method = str(infer_cfg.method)
        self._posterior_state = _PosteriorState(method=method)
        fsps_age_grid = host_cfg.age_grid_gyr
        fsps_logzsol_grid = host_cfg.logzsol_grid
        host_sfh_model = str(host_cfg.sfh_model)
        dsps_ssp_fn = host_cfg.dsps_ssp_fn
        nuts_warmup = int(infer_cfg.num_warmup)
        nuts_samples = int(infer_cfg.num_samples)
        nuts_chains = int(infer_cfg.num_chains)
        nuts_target_accept = float(infer_cfg.target_accept_prob)
        optax_steps = int(infer_cfg.map_steps)
        optax_lr = float(infer_cfg.learning_rate)
        plot_init = bool(infer_cfg.plot_init)
        prior_config = None if cfg.prior_config is None else _materialize_prior_config(cfg.prior_config)
        if psf_cfg is not None:
            psf_mags = psf_cfg.magnitudes
            psf_mag_errs = psf_cfg.magnitude_errors
            psf_bands = psf_cfg.filter_names
        else:
            psf_mags = None
            psf_mag_errs = None
            psf_bands = None
        use_psf_phot = bool(psf_cfg is not None)

        save_result = bool(out_cfg.save_result)
        plot_fig = bool(out_cfg.plot_fig)
        save_fig = bool(out_cfg.save_fig)
        show_plot = bool(out_cfg.show_plot)
        if self.output_path is None and out_cfg.output_path is not None:
            self.output_path = out_cfg.output_path
        custom_components = line_cfg.custom_components
        custom_line_components = line_cfg.custom_line_components

        if kwargs_plot is None:
            kwargs_plot = {}
        if 'show_plot' not in kwargs_plot:
            kwargs_plot['show_plot'] = show_plot

        # Persist fit configuration so posterior reconstructions can be built on
        # alternate wavelength grids after fitting.
        self._fit_deredden = bool(deredden)
        self._fit_decompose_host = bool(decompose_host)
        self._fit_fit_lines = bool(fit_lines)
        self._fit_fit_pl = bool(fit_pl)
        self._fit_fit_fe = bool(fit_fe)
        self._fit_fit_bc = bool(fit_bc)
        self._fit_fit_bal = bool(fit_bal)
        self._fit_fit_poly = bool(fit_poly)
        self._fit_fit_reddening = bool(fit_reddening)
        self._fit_fit_poly_order = int(fit_poly_order)
        self._fit_mask_lya_forest = bool(mask_lya_forest)
        self._fit_inference_method = str(method)
        self._fit_fsps_age_grid = tuple(fsps_age_grid)
        self._fit_fsps_logzsol_grid = tuple(fsps_logzsol_grid)
        self._fit_host_sfh_model = str(host_sfh_model)
        self._fit_prior_config = prior_config
        self._fit_dsps_ssp_fn = str(dsps_ssp_fn)
        self._fit_use_psf_phot = bool(use_psf_phot)
        requested_custom_components = normalize_custom_components(custom_components)
        self._fit_custom_components = requested_custom_components
        self._fit_custom_line_components = normalize_custom_line_components(custom_line_components)

        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.linefit = fit_lines
        self.save_fig = save_fig
        self.verbose = verbose
        if name is not None and str(name).strip() != "":
            self.filename = str(name).strip()
        prior_config_input = prior_config
        prior_config = {} if prior_config is None else prior_config

        data_dir = os.path.join(self.install_path, 'data')
        self.fe_uv = np.genfromtxt(os.path.join(data_dir, 'fe_uv.txt'))
        self.fe_op = np.genfromtxt(os.path.join(data_dir, 'fe_optical.txt'))

        self.fe_uv_wave = 10 ** self.fe_uv[:, 0]
        # Normalize non-negative template amplitudes to O(1) so Fe norms are in data-flux units.
        self.fe_uv_flux = _normalize_template_flux(np.maximum(self.fe_uv[:, 1], 0.0), target_amp=1.0)

        fe_op_wave = 10 ** self.fe_op[:, 0]
        fe_op_flux = _normalize_template_flux(np.maximum(self.fe_op[:, 1], 0.0), target_amp=1.0)
        m = (fe_op_wave > 3686.) & (fe_op_wave < 7484.)
        self.fe_op_wave = fe_op_wave[m]
        self.fe_op_flux = fe_op_flux[m]

        save_fits_name = self.filename

        ind_gooderror = np.where((self.err_in > 0) & np.isfinite(self.err_in) & (self.flux_in != 0) & np.isfinite(self.flux_in), True, False)
        self.err = self.err_in[ind_gooderror]
        self.flux = self.flux_in[ind_gooderror]
        self.lam = self.lam_in[ind_gooderror]

        if wave_range is not None:
            self._wave_trim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._wave_msk(self.lam, self.flux, self.err, self.z)
        if mask_lya_forest:
            self._mask_lya_forest(self.lam, self.flux, self.err, self.z)
        if deredden:
            self._validate_deredden_coordinates(self.ra, self.dec)
            self._de_redden(self.lam, self.flux, self.err, self.ra, self.dec)

        self._rest_frame(self.lam, self.flux, self.err, self.z)
        self._calculate_sn(self.wave, self.flux)
        self._orignial_spec(self.wave, self.flux, self.err)

        bal_components = build_default_bal_components(self.flux) if bool(fit_bal) else ()
        self._fit_custom_components = normalize_custom_components(
            tuple(requested_custom_components) + tuple(bal_components)
        )

        if prior_config_input is None:
            prior_config = _materialize_prior_config(build_default_prior_config(self.flux))
        prior_config["z_qso"] = float(self.z)
        prior_config["host_sfh_model"] = str(host_sfh_model)
        self._fit_host_sfh_model = str(prior_config.get("host_sfh_model", "flexible"))
        prior_config = inject_default_custom_component_priors(
            prior_config=prior_config,
            flux=self.flux,
            custom_components=self._fit_custom_components,
        )
        prior_config = inject_default_custom_line_component_priors(
            prior_config=prior_config,
            flux=self.flux,
            custom_line_components=self._fit_custom_line_components,
        )
        out_params = prior_config.get('out_params', {})
        self.L_conti_wave = np.asarray(out_params.get('cont_loc', []), dtype=float)
        self._fit_prior_config = prior_config

        pl_pivot = prior_config.get("PL_pivot", None)
        if pl_pivot is None:
            pl_pivot = _spectrum_center_pivot(self.wave)
        prior_config["PL_pivot"] = float(np.asarray(pl_pivot, dtype=float))
        poly_pivot = prior_config.get("poly_pivot", None)
        if poly_pivot is None:
            poly_pivot = _spectrum_center_pivot(self.wave)
        prior_config["poly_pivot"] = float(np.asarray(poly_pivot, dtype=float))
        self._fit_prior_config = prior_config
        psf_mags_use, psf_mag_errs_use, _psf_bands_use, psf_filter_curves_use, use_psf_phot_use = self._prepare_psf_photometry(
            wave_obs=self.lam,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_bands=psf_bands,
            use_psf_phot=use_psf_phot,
        )

        if method == 'nuts':
            self.run_fsps_numpyro_fit(
                num_warmup=nuts_warmup,
                num_samples=nuts_samples,
                num_chains=nuts_chains,
                target_accept_prob=nuts_target_accept,
                age_grid_gyr=fsps_age_grid,
                logzsol_grid=fsps_logzsol_grid,
                prior_config=prior_config,
                dsps_ssp_fn=dsps_ssp_fn,
                use_lines=fit_lines,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=fit_fe,
                fit_bc=fit_bc,
                fit_poly=fit_poly,
                fit_reddening=fit_reddening,
                fit_poly_order=fit_poly_order,
                psf_mags=psf_mags_use,
                psf_mag_errs=psf_mag_errs_use,
                psf_filter_curves=psf_filter_curves_use,
                use_psf_phot=use_psf_phot_use,
                custom_components=self._fit_custom_components,
                custom_line_components=self._fit_custom_line_components,
            )
        elif method == 'optax':
            self.run_fsps_optax_fit(
                num_steps=optax_steps,
                learning_rate=optax_lr,
                age_grid_gyr=fsps_age_grid,
                logzsol_grid=fsps_logzsol_grid,
                prior_config=prior_config,
                dsps_ssp_fn=dsps_ssp_fn,
                use_lines=fit_lines,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=fit_fe,
                fit_bc=fit_bc,
                fit_poly=fit_poly,
                fit_reddening=fit_reddening,
                fit_poly_order=fit_poly_order,
                psf_mags=psf_mags_use,
                psf_mag_errs=psf_mag_errs_use,
                psf_filter_curves=psf_filter_curves_use,
                use_psf_phot=use_psf_phot_use,
                custom_components=self._fit_custom_components,
                custom_line_components=self._fit_custom_line_components,
                plot_init=plot_init,
            )
        elif method == 'optax+nuts':
            self.run_fsps_optax_nuts_fit(
                optax_steps=optax_steps,
                optax_learning_rate=optax_lr,
                num_warmup=nuts_warmup,
                num_samples=nuts_samples,
                num_chains=nuts_chains,
                target_accept_prob=nuts_target_accept,
                age_grid_gyr=fsps_age_grid,
                logzsol_grid=fsps_logzsol_grid,
                prior_config=prior_config,
                dsps_ssp_fn=dsps_ssp_fn,
                use_lines=fit_lines,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=fit_fe,
                fit_bc=fit_bc,
                fit_poly=fit_poly,
                fit_reddening=fit_reddening,
                fit_poly_order=fit_poly_order,
                psf_mags=psf_mags_use,
                psf_mag_errs=psf_mag_errs_use,
                psf_filter_curves=psf_filter_curves_use,
                use_psf_phot=use_psf_phot_use,
                custom_components=self._fit_custom_components,
                custom_line_components=self._fit_custom_line_components,
                plot_init=plot_init,
            )
        else:
            raise ValueError(f"Unknown inference method='{method}'. Use 'nuts', 'optax', or 'optax+nuts'.")

        posterior_bundle_path = None
        if save_result:
            self.save_result(self.conti_result, self.conti_result_type, self.conti_result_name,
                             self.line_result, self.line_result_type, self.line_result_name,
                             save_fits_name)
            posterior_bundle_path = self.save_posterior_bundle()
        if plot_fig:
            self.plot_fig(**kwargs_plot)
        return self._make_result(
            method=method,
            path=posterior_bundle_path,
            figure=getattr(self, "fig", None),
        )

    def run_fsps_numpyro_fit(self, num_warmup=500, num_samples=1000, num_chains=1,
                             target_accept_prob=0.9,
                             age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                             logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                             prior_config=None,
                             dsps_ssp_fn='tempdata.h5',
                             use_lines=True,
                             decompose_host=True,
                             fit_pl=True,
                             fit_fe=True,
                             fit_bc=True,
                             fit_poly=False,
                             fit_reddening=False,
                             fit_poly_order=2,
                             psf_mags=None,
                             psf_mag_errs=None,
                             psf_filter_curves=None,
                             use_psf_phot=False,
                             custom_components=None,
                             custom_line_components=None,
                             init_values=None):
        """Fit the full model using NUTS MCMC and store posterior summaries.

        Parameters
        ----------
        num_warmup, num_samples : int, optional
            MCMC warmup and posterior sample counts.
        num_chains : int, optional
            Number of MCMC chains.
        target_accept_prob : float, optional
            Target acceptance probability for NUTS.
        age_grid_gyr : sequence of float, optional
            SSP age grid in Gyr.
        logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary for model blocks.
        dsps_ssp_fn : str, optional
            DSPS SSP template HDF5 path.
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_reddening : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        init_values : dict or None, optional
            Optional initial values for ``init_to_value``.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        custom_components = normalize_custom_components(custom_components)
        custom_line_components = normalize_custom_line_components(custom_line_components)
        if prior_config is None:
            prior_config = _materialize_prior_config(build_default_prior_config(flux))
        prior_config = inject_default_custom_component_priors(prior_config, flux, custom_components)
        prior_config = inject_default_custom_line_component_priors(prior_config, flux, custom_line_components)
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None and len(custom_line_components) == 0:
            raise ValueError(
                "fit_lines=True requires either line priors/table in prior_config "
                "or at least one custom_line_component."
            )

        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)
        else:
            tied_line_meta = {
                'n_lines': 0,
                'n_vgroups': 0,
                'n_wgroups': 0,
                'n_fgroups': 0,
                'ln_lambda0': _np_to_jnp(np.array([], dtype=float)),
                'vgroup': np.array([], dtype=int),
                'wgroup': np.array([], dtype=int),
                'fgroup': np.array([], dtype=int),
                'flux_ratio': np.array([], dtype=float),
                'dmu_init_group': np.array([], dtype=float),
                'dmu_min_group': np.array([], dtype=float),
                'dmu_max_group': np.array([], dtype=float),
                'sig_init_group': np.array([], dtype=float),
                'sig_min_group': np.array([], dtype=float),
                'sig_max_group': np.array([], dtype=float),
                'amp_init_group': np.array([], dtype=float),
                'amp_min_group': np.array([], dtype=float),
                'amp_max_group': np.array([], dtype=float),
                'names': [],
                'compnames': [],
                'line_lambda': np.array([], dtype=float),
            }
        fsps_grid = self._build_fsps_grid_for_fit(
            wave=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
            decompose_host=decompose_host,
            z_qso=self.z,
        )
        self.tied_line_meta = tied_line_meta

        if init_values is None:
            init_vals = {
                'gal_v_kms': 0.0,
                'log_gal_sigma_kms': np.log(150.0),
            }
            host_sfh_model = str(prior_config.get("host_sfh_model", "flexible")).lower()
            if decompose_host and not (
                host_sfh_model in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}
                and getattr(fsps_grid, "host_basis_jax", None) is not None
            ):
                init_vals['cont_norm'] = np.exp(prior_config.get('log_cont_norm', {}).get('loc', np.log(max(np.nanmedian(np.abs(flux)), 1e-8))))
                init_vals['log_frac_host'] = prior_config.get('log_frac_host', {}).get('loc', 0.0)
            if fit_reddening:
                init_vals['log_reddening_a2500'] = prior_config.get('log_reddening_a2500', {}).get('loc', np.log(0.1))
        else:
            init_vals = init_values
        init_strategy = init_to_value(values=init_vals)
        kernel = NUTS(qso_fsps_joint_model, init_strategy=init_strategy, target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=8)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True, jit_model_args=False)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(
            rng_key,
            wave=wave,
            flux=flux,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_reddening=fit_reddening,
            fit_poly_order=fit_poly_order,
            z_qso=self.z,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
            return_line_components=False,
            emit_deterministics=False,
            custom_components=custom_components,
            custom_line_components=custom_line_components,
        )
        samples = mcmc.get_samples()

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples=samples,
            return_sites=self._predictive_return_sites(custom_components=custom_components, custom_line_components=custom_line_components),
        )
        pred_out = pred(
            rng_key,
            wave=wave,
            flux=None,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_reddening=fit_reddening,
            fit_poly_order=fit_poly_order,
            z_qso=self.z,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
            custom_components=custom_components,
            custom_line_components=custom_line_components,
        )

        self.numpyro_mcmc = mcmc
        self._consume_posterior_outputs(
            samples=samples,
            pred_out=pred_out,
            fsps_grid=fsps_grid,
            tied_line_meta=tied_line_meta,
            use_lines=use_lines,
            decompose_host=decompose_host,
        )

    def _plot_stage1_initialization(self, wave, flux, err, pred_out, samples):
        """Plot and store the stage-1 Optax continuum/host warm-start model."""
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        err = np.asarray(err, dtype=float)
        model = np.median(np.asarray(pred_out['model']), axis=0)
        host = np.median(np.asarray(pred_out['gal_model']), axis=0)
        pl = np.median(np.asarray(pred_out['f_pl_model']), axis=0)
        line = np.median(np.asarray(pred_out['line_model']), axis=0)
        continuum = np.median(np.asarray(pred_out['continuum_model']), axis=0)

        valid = (
            np.isfinite(wave)
            & np.isfinite(flux)
            & np.isfinite(err)
            & np.isfinite(model)
            & (err > 0)
        )
        n_params = len(samples)
        dof = max(int(np.sum(valid)) - n_params, 1)
        redchi2 = float(np.sum(((flux[valid] - model[valid]) / err[valid]) ** 2) / dof)

        self.init_stage1_samples = samples
        self.init_stage1_pred_out = pred_out
        self.init_stage1_model = model
        self.init_stage1_continuum_model = continuum
        self.init_stage1_host_model = host
        self.init_stage1_pl_model = pl
        self.init_stage1_line_model = line
        self.init_stage1_redchi2 = redchi2

        fig, (ax, axr) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12, 6),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
        )
        ax.plot(wave, flux, color='black', lw=0.8, alpha=0.8, label='data')
        ax.plot(wave, model, color='blue', lw=1.6, label='stage 1 model')
        ax.plot(wave, host, color='purple', lw=1.2, label='host galaxy')
        ax.plot(wave, pl, color='orange', lw=1.2, label='power law')
        if np.nanmax(np.abs(line)) > 0:
            ax.plot(wave, line, color='lightskyblue', lw=1.0, label='lines')
        ax.set_ylabel(r'$f_\lambda$')
        ax.set_title(f'Stage 1 initialization (reduced chi2 = {redchi2:.2f})')
        ax.legend(loc='best')

        resid = flux - model
        axr.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.6)
        axr.plot(wave, resid, color='gray', lw=0.8, ls=':', alpha=0.9)
        axr.set_ylabel('resid')
        axr.set_xlabel(r'Rest Wavelength ($\AA$)')
        plt.show()

    def run_fsps_optax_fit(self, num_steps=2000, learning_rate=1e-2,
                           age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                           logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                           prior_config=None,
                           dsps_ssp_fn='tempdata.h5',
                           use_lines=True,
                           decompose_host=True,
                           fit_pl=True,
                           fit_fe=True,
                           fit_bc=True,
                           fit_poly=False,
                           fit_reddening=False,
                           fit_poly_order=2,
                           psf_mags=None,
                           psf_mag_errs=None,
                           psf_filter_curves=None,
                           use_psf_phot=False,
                           custom_components=None,
                           custom_line_components=None,
                           plot_init=False):
        """Fit a MAP approximation using staged SVI with an Optax optimizer.

        Parameters
        ----------
        num_steps : int, optional
            Total SVI steps across all stages.
        learning_rate : float, optional
            Adam learning rate.
        age_grid_gyr : sequence of float, optional
            SSP age grid in Gyr.
        logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary for model blocks.
        dsps_ssp_fn : str, optional
            DSPS SSP template HDF5 path.
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_reddening : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        plot_init : bool, optional
            If True, plot and store the stage-1 continuum/host warm-start
            model before starting the full model stage.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        custom_components = normalize_custom_components(custom_components)
        custom_line_components = normalize_custom_line_components(custom_line_components)
        if prior_config is None:
            prior_config = _materialize_prior_config(build_default_prior_config(flux))
        prior_config = inject_default_custom_component_priors(prior_config, flux, custom_components)
        prior_config = inject_default_custom_line_component_priors(prior_config, flux, custom_line_components)
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None and len(custom_line_components) == 0:
            raise ValueError(
                "fit_lines=True requires either line priors/table in prior_config "
                "or at least one custom_line_component."
            )

        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)
        else:
            tied_line_meta = {
                'n_lines': 0,
                'n_vgroups': 0,
                'n_wgroups': 0,
                'n_fgroups': 0,
                'ln_lambda0': _np_to_jnp(np.array([], dtype=float)),
                'vgroup': np.array([], dtype=int),
                'wgroup': np.array([], dtype=int),
                'fgroup': np.array([], dtype=int),
                'flux_ratio': np.array([], dtype=float),
                'dmu_init_group': np.array([], dtype=float),
                'dmu_min_group': np.array([], dtype=float),
                'dmu_max_group': np.array([], dtype=float),
                'sig_init_group': np.array([], dtype=float),
                'sig_min_group': np.array([], dtype=float),
                'sig_max_group': np.array([], dtype=float),
                'amp_init_group': np.array([], dtype=float),
                'amp_min_group': np.array([], dtype=float),
                'amp_max_group': np.array([], dtype=float),
                'names': [],
                'compnames': [],
                'line_lambda': np.array([], dtype=float),
            }
        fsps_grid = self._build_fsps_grid_for_fit(
            wave=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
            decompose_host=decompose_host,
            z_qso=self.z,
        )
        self.tied_line_meta = tied_line_meta

        def _subset_fsps_grid(grid, keep_mask):
            """Return an FSPS grid restricted to the selected wavelength pixels."""
            keep_mask = np.asarray(keep_mask, dtype=bool)
            host_basis = getattr(grid, "host_basis_jax", None)
            if host_basis is not None:
                host_basis = replace(
                    host_basis,
                    rest_llambda=host_basis.rest_llambda[..., keep_mask],
                )
            return FSPSTemplateGrid(
                wave=np.asarray(grid.wave)[keep_mask],
                templates=np.asarray(grid.templates)[keep_mask, :],
                template_meta=grid.template_meta,
                age_grid_gyr=grid.age_grid_gyr,
                logzsol_grid=grid.logzsol_grid,
                host_basis_jax=host_basis,
                t_obs_gyr=grid.t_obs_gyr,
            )

        def _stage1_continuum_keep_mask(wave_in):
            """Mask strong optical emission-line windows for continuum warm start."""
            line_windows = (
                (3700.0, 3755.0),   # [O II]
                (3850.0, 3895.0),   # [Ne III]
                (4070.0, 4135.0),   # Hdelta
                (4300.0, 4385.0),   # Hgamma + [O III] 4363
                (4630.0, 5105.0),   # He II, Hbeta, [O III]
                (5800.0, 5925.0),   # He I
                (6250.0, 6405.0),   # [O I]
                (6450.0, 6775.0),   # Halpha, [N II], [S II]
                (7050.0, 7165.0),   # He I / [Ar III]
                (7300.0, 7355.0),   # [O II]
            )
            wave_in = np.asarray(wave_in, dtype=float)
            keep = np.isfinite(wave_in)
            for lo, hi in line_windows:
                keep &= ~((wave_in >= lo) & (wave_in <= hi))
            min_keep = max(50, int(0.2 * wave_in.size))
            if int(np.sum(keep)) < min_keep:
                return np.isfinite(wave_in)
            return keep

        def _subset_psf_filter_curves(curves, keep_mask):
            """Return PSF filter curves restricted to a wavelength subset."""
            if curves is None:
                return None
            subset = dict(curves)
            if "trans" in subset:
                subset["trans"] = np.asarray(subset["trans"])[..., np.asarray(keep_mask, dtype=bool)]
            return subset

        def _run_svi(
            guide,
            steps,
            use_lines_i,
            fit_pl_i,
            fit_fe_i,
            fit_bc_i,
            fit_poly_i,
            fit_reddening_i,
            fit_poly_order_i,
            decompose_host_i,
            wave_i=None,
            flux_i=None,
            err_i=None,
            fsps_grid_i=None,
            psf_filter_curves_i=None,
        ):
            """Run an SVI stage and return optimizer state/results."""
            wave_run = wave if wave_i is None else wave_i
            flux_run = flux if flux_i is None else flux_i
            err_run = err if err_i is None else err_i
            fsps_grid_run = fsps_grid if fsps_grid_i is None else fsps_grid_i
            psf_filter_curves_run = psf_filter_curves if psf_filter_curves_i is None else psf_filter_curves_i
            optimizer = optax_to_numpyro(optax.adam(learning_rate))
            svi = SVI(qso_fsps_joint_model, guide, optimizer, loss=Trace_ELBO())
            key = jax.random.PRNGKey(0)
            result = svi.run(
                key,
                int(steps),
                wave=wave_run,
                flux=flux_run,
                err=err_run,
                conti_priors=conti_priors,
                tied_line_meta=tied_line_meta,
                fsps_grid=fsps_grid_run,
                fe_uv_wave=self.fe_uv_wave,
                fe_uv_flux=self.fe_uv_flux,
                fe_op_wave=self.fe_op_wave,
                fe_op_flux=self.fe_op_flux,
                use_lines=use_lines_i,
                prior_config=prior_config,
                decompose_host=decompose_host_i,
                fit_pl=fit_pl_i,
                fit_fe=fit_fe_i,
                fit_bc=fit_bc_i,
                fit_poly=fit_poly_i,
                fit_reddening=fit_reddening_i,
                fit_poly_order=fit_poly_order_i,
                z_qso=self.z,
                psf_mags=psf_mags,
                psf_mag_errs=psf_mag_errs,
                psf_filter_curves=psf_filter_curves_run,
                use_psf_phot=use_psf_phot,
                return_line_components=False,
                emit_deterministics=False,
                custom_components=custom_components,
                custom_line_components=custom_line_components,
                progress_bar=self.verbose,
            )
            return svi, result

        def _prior_field(key, field, default):
            """Read a scalar field from a prior-config entry."""
            cfg = prior_config.get(key, default)
            if isinstance(cfg, dict):
                value = cfg.get(field, cfg.get('value', cfg.get('loc', default)))
            elif isinstance(cfg, (tuple, list)) and len(cfg) > 0:
                value = cfg[0]
            else:
                value = cfg
            try:
                value = float(np.asarray(value, dtype=float))
            except Exception:
                value = float(default)
            return value if np.isfinite(value) else float(default)

        def _stage1_init_values():
            """Build data-scale-aware constrained initial values for stage 1."""
            pl_init = _prior_field('PL_norm', 'scale', max(0.5 * np.nanmedian(np.abs(flux)), 1e-8))
            host_sfh_model = str(prior_config.get("host_sfh_model", "flexible")).lower()
            use_direct_host_amp = not (
                decompose_host
                and host_sfh_model in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}
                and getattr(fsps_grid, "host_basis_jax", None) is not None
            )

            values = {
                'gal_v_kms': 0.0,
                'log_gal_sigma_kms': _prior_field('log_gal_sigma_kms', 'loc', np.log(150.0)),
            }
            if decompose_host and use_direct_host_amp:
                values['cont_norm'] = max(np.exp(_prior_field('log_cont_norm', 'loc', np.log(max(np.nanmedian(np.abs(flux)), 1e-8)))), 1e-8)
                values['log_frac_host'] = _prior_field('log_frac_host', 'loc', 0.0)
            if fit_pl:
                values['PL_norm'] = max(pl_init, 1e-8)
                if fit_reddening:
                    values['log_reddening_a2500'] = _prior_field('log_reddening_a2500', 'loc', np.log(0.1))

            if decompose_host and host_sfh_model in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}:
                values['log_stellar_mass'] = _prior_field('log_stellar_mass', 'loc', 9.0)
                values['log_sfh_age_gyr'] = _prior_field('log_sfh_age_gyr', 'loc', np.log(3.0))
                values['log_sfh_tau_over_age'] = _prior_field('log_sfh_tau_over_age', 'loc', 0.0)
                values['gal_lgmet'] = _prior_field('gal_lgmet', 'loc', 0.0)
                values['log_gal_lgmet_scatter'] = _prior_field('log_gal_lgmet_scatter', 'loc', np.log(0.15))
                values['log_host_aperture_scale'] = _prior_field('log_host_aperture_scale', 'value', 0.0)
            return values

        # Stage 1: warm start on simpler landscape (continuum/host only).
        n1 = max(100, int(num_steps // 3))
        stage1_keep = _stage1_continuum_keep_mask(wave)
        self.init_stage1_keep_mask = stage1_keep
        fsps_grid_stage1 = _subset_fsps_grid(fsps_grid, stage1_keep)
        psf_filter_curves_stage1 = _subset_psf_filter_curves(psf_filter_curves, stage1_keep)
        stage1_init_values = _stage1_init_values()
        guide1 = AutoDelta(
            qso_fsps_joint_model,
            init_loc_fn=init_to_value(values=stage1_init_values),
        )
        svi1, res1 = _run_svi(
            guide1,
            n1,
            use_lines_i=False,
            fit_pl_i=fit_pl,
            fit_fe_i=False,
            fit_bc_i=False,
            fit_poly_i=False,
            fit_reddening_i=False,
            fit_poly_order_i=2,
            decompose_host_i=decompose_host,
            wave_i=wave[stage1_keep],
            flux_i=flux[stage1_keep],
            err_i=err[stage1_keep],
            fsps_grid_i=fsps_grid_stage1,
            psf_filter_curves_i=psf_filter_curves_stage1,
        )
        map1 = guide1.median(res1.params)
        if plot_init:
            stage1_samples = {k: np.asarray(v)[None, ...] for k, v in map1.items()}
            pred1 = Predictive(
                qso_fsps_joint_model,
                posterior_samples={k: jnp.asarray(v) for k, v in stage1_samples.items()},
                return_sites=[
                    'f_pl_model',
                    'gal_model',
                    'line_model',
                    'continuum_model',
                    'model',
                ],
            )
            pred1_out = pred1(
                jax.random.PRNGKey(1),
                wave=wave,
                flux=None,
                err=err,
                conti_priors=conti_priors,
                tied_line_meta=tied_line_meta,
                fsps_grid=fsps_grid,
                fe_uv_wave=self.fe_uv_wave,
                fe_uv_flux=self.fe_uv_flux,
                fe_op_wave=self.fe_op_wave,
                fe_op_flux=self.fe_op_flux,
                use_lines=False,
                prior_config=prior_config,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=False,
                fit_bc=False,
                fit_poly=False,
                fit_reddening=False,
                fit_poly_order=2,
                z_qso=self.z,
                psf_mags=psf_mags,
                psf_mag_errs=psf_mag_errs,
                psf_filter_curves=psf_filter_curves,
                use_psf_phot=use_psf_phot,
                custom_components=custom_components,
                custom_line_components=custom_line_components,
            )
            self._plot_stage1_initialization(
                wave=wave,
                flux=flux,
                err=err,
                pred_out=pred1_out,
                samples=stage1_samples,
            )

        # Stage 2: full model initialized from stage-1 MAP for overlapping parameters.
        n2 = max(100, int(num_steps - n1))
        guide2 = AutoDelta(
            qso_fsps_joint_model,
            init_loc_fn=init_to_value(values=map1),
        )
        svi, res2 = _run_svi(
            guide2,
            n2,
            use_lines_i=use_lines,
            fit_pl_i=fit_pl,
            fit_fe_i=fit_fe,
            fit_bc_i=fit_bc,
            fit_poly_i=fit_poly,
            fit_reddening_i=fit_reddening,
            fit_poly_order_i=fit_poly_order,
            decompose_host_i=decompose_host,
        )

        svi_state = res2.state
        svi_params = res2.params
        losses = np.concatenate([np.asarray(res1.losses), np.asarray(res2.losses)])
        map_point = guide2.median(svi_params)
        samples = {k: np.asarray(v)[None, ...] for k, v in map_point.items()}
        rng_key = jax.random.PRNGKey(0)

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples={k: jnp.asarray(v) for k, v in samples.items()},
            return_sites=self._predictive_return_sites(custom_components=custom_components, custom_line_components=custom_line_components),
        )
        pred_out = pred(
            rng_key,
            wave=wave,
            flux=None,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_reddening=fit_reddening,
            fit_poly_order=fit_poly_order,
            z_qso=self.z,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
            custom_components=custom_components,
            custom_line_components=custom_line_components,
        )

        self.numpyro_mcmc = None
        self.svi = svi
        self.svi_state = svi_state
        self.svi_params = svi_params
        self.optax_losses = losses
        self.optax_map_point = map_point
        self._consume_posterior_outputs(
            samples=samples,
            pred_out=pred_out,
            fsps_grid=fsps_grid,
            tied_line_meta=tied_line_meta,
            use_lines=use_lines,
            decompose_host=decompose_host,
        )

    def run_fsps_optax_nuts_fit(self, optax_steps=2000, optax_learning_rate=1e-2,
                                num_warmup=500, num_samples=1000, num_chains=1,
                                target_accept_prob=0.9,
                                age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                                logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                                prior_config=None,
                                dsps_ssp_fn='tempdata.h5',
                                use_lines=True,
                                decompose_host=True,
                                fit_pl=True,
                                fit_fe=True,
                                fit_bc=True,
                                fit_poly=False,
                                fit_reddening=False,
                                fit_poly_order=2,
                                psf_mags=None,
                                psf_mag_errs=None,
                                psf_filter_curves=None,
                                use_psf_phot=False,
                                custom_components=None,
                                custom_line_components=None,
                                plot_init=False):
        """Warm-start with Optax MAP, then run NUTS as final inference.

        Parameters
        ----------
        optax_steps : int, optional
            Number of SVI/Optax warm-start steps.
        optax_learning_rate : float, optional
            Learning rate for SVI warm-start.
        num_warmup, num_samples : int, optional
            NUTS warmup and posterior sample counts.
        num_chains : int, optional
            Number of MCMC chains.
        target_accept_prob : float, optional
            Target acceptance probability for NUTS.
        age_grid_gyr : sequence of float, optional
            SSP age grid in Gyr.
        logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary for model blocks.
        dsps_ssp_fn : str, optional
            DSPS SSP template HDF5 path.
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_reddening : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        plot_init : bool, optional
            If True, plot and store the stage-1 Optax warm-start model before
            starting the full Optax stage and NUTS.
        """
        self.run_fsps_optax_fit(
            num_steps=optax_steps,
            learning_rate=optax_learning_rate,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            prior_config=prior_config,
            dsps_ssp_fn=dsps_ssp_fn,
            use_lines=use_lines,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_reddening=fit_reddening,
            fit_poly_order=fit_poly_order,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
            custom_components=custom_components,
            custom_line_components=custom_line_components,
            plot_init=plot_init,
        )
        init_values = getattr(self, 'optax_map_point', None)
        self.run_fsps_numpyro_fit(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            prior_config=prior_config,
            dsps_ssp_fn=dsps_ssp_fn,
            use_lines=use_lines,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_reddening=fit_reddening,
            fit_poly_order=fit_poly_order,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
            custom_components=custom_components,
            custom_line_components=custom_line_components,
            init_values=init_values,
        )

    def _consume_posterior_outputs(self, samples, pred_out, fsps_grid, tied_line_meta, use_lines, decompose_host):
        """Populate model components, uncertainty bands, and summary tables.

        Parameters
        ----------
        samples : dict
            Posterior samples keyed by parameter name.
        pred_out : dict
            Posterior predictive outputs from ``Predictive``.
        fsps_grid : FSPSTemplateGrid
            Host SSP template grid metadata.
        tied_line_meta : dict
            Emission-line grouping metadata.
        use_lines : bool
            Whether line model was enabled.
        decompose_host : bool
            Whether host model was enabled.
        """
        flux = np.asarray(self.flux, dtype=float)
        self.numpyro_samples = samples
        self.fsps_grid = fsps_grid
        self._fit_fsps_template_norms = tuple(
            float(meta.get("norm", 1.0)) for meta in getattr(fsps_grid, "template_meta", [])
        )
        self.pred_out = pred_out
        self._pred_host_draws = np.asarray(pred_out['gal_model'])
        self._pred_bc_draws = np.asarray(pred_out['f_bc_model'])
        self._pred_cont_draws = np.asarray(pred_out['continuum_model'])
        self._pred_total_draws = np.asarray(pred_out['model'])
        self._pred_line_draws = np.asarray(pred_out['line_model'])
        self._pred_psf_draws = np.asarray(pred_out['psf_model']) if 'psf_model' in pred_out else None
        self.custom_components = {}
        self._pred_custom_draws = {}
        self.custom_line_components = {}
        self._pred_custom_line_draws = {}
        self.bi = np.nan
        self.bi_err = np.nan

        self.f_pl_model = np.median(np.asarray(pred_out['f_pl_model']), axis=0)
        intrinsic_pl_draws = self._intrinsic_powerlaw_draws()
        if intrinsic_pl_draws is not None and intrinsic_pl_draws.shape[1] == len(self.wave):
            self.f_pl_model_intrinsic = np.median(intrinsic_pl_draws, axis=0)
        else:
            self.f_pl_model_intrinsic = np.full_like(self.f_pl_model, np.nan)
        self.f_fe_mgii_model = np.median(np.asarray(pred_out['f_fe_mgii_model']), axis=0)
        self.f_fe_balmer_model = np.median(np.asarray(pred_out['f_fe_balmer_model']), axis=0)
        self.f_bc_model = np.median(np.asarray(pred_out['f_bc_model']), axis=0)
        self.f_poly_model = np.median(np.asarray(pred_out['f_poly_model']), axis=0)
        self.qso = np.median(np.asarray(pred_out['agn_model']), axis=0)
        self.host = np.median(np.asarray(pred_out['gal_model']), axis=0)
        self.line_broad = np.median(np.asarray(pred_out['line_model_broad']), axis=0) if 'line_model_broad' in pred_out else np.full_like(self.qso, np.nan)
        self.line_narrow = np.median(np.asarray(pred_out['line_model_narrow']), axis=0) if 'line_model_narrow' in pred_out else np.full_like(self.qso, np.nan)
        self.line_component_profiles = np.median(np.asarray(pred_out['line_component_profiles']), axis=0) if 'line_component_profiles' in pred_out else np.empty((0, len(self.wave)), dtype=float)
        self.f_line_model = np.median(np.asarray(pred_out['line_model']), axis=0)
        self.f_conti_model = np.median(np.asarray(pred_out['continuum_model']), axis=0)
        self.model_total = np.median(np.asarray(pred_out['model']), axis=0)
        self.qso_psf = np.median(np.asarray(pred_out['agn_model_psf']), axis=0) if 'agn_model_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.host_psf = np.median(np.asarray(pred_out['gal_model_psf']), axis=0) if 'gal_model_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.line_broad_psf = np.median(np.asarray(pred_out['line_model_broad_psf']), axis=0) if 'line_model_broad_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.line_narrow_psf = np.median(np.asarray(pred_out['line_model_narrow_psf']), axis=0) if 'line_model_narrow_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.line_component_profiles_psf = np.median(np.asarray(pred_out['line_component_profiles_psf']), axis=0) if 'line_component_profiles_psf' in pred_out else np.empty((0, len(self.wave)), dtype=float)
        self.line_psf = np.median(np.asarray(pred_out['line_model_psf']), axis=0) if 'line_model_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.psf_model = np.median(np.asarray(pred_out['psf_model']), axis=0) if 'psf_model' in pred_out else np.full_like(self.model_total, np.nan)
        self.fsps_weights_median = np.median(np.asarray(pred_out['fsps_weights']), axis=0)
        for comp in normalize_custom_components(getattr(self, '_fit_custom_components', ())):
            if comp.deterministic_site_name in pred_out:
                draws = np.asarray(pred_out[comp.deterministic_site_name])
                self._pred_custom_draws[comp.output_name] = draws
                self.custom_components[comp.output_name] = np.median(draws, axis=0)
        for comp in normalize_custom_line_components(getattr(self, '_fit_custom_line_components', ())):
            if comp.deterministic_site_name in pred_out:
                draws = np.asarray(pred_out[comp.deterministic_site_name])
                self._pred_custom_line_draws[comp.output_name] = draws
                self.custom_line_components[comp.output_name] = np.median(draws, axis=0)
        self.line_flux = flux - self.f_conti_model
        self.decomposed = True
        if 'delta_m_psf_raw' in samples:
            delta_m_draws = np.asarray(samples['delta_m_psf_raw'], dtype=float)
        elif 'delta_m_psf' in pred_out:
            delta_m_draws = np.asarray(pred_out['delta_m_psf'], dtype=float)
        else:
            delta_m_draws = np.array([np.nan], dtype=float)
        if 'eta_psf_raw' in samples:
            eta_psf_draws = np.asarray(samples['eta_psf_raw'], dtype=float)
        elif 'eta_psf' in pred_out:
            eta_psf_draws = np.asarray(pred_out['eta_psf'], dtype=float)
        else:
            eta_psf_draws = np.array([np.nan], dtype=float)
        self.delta_m_psf = float(np.nanmedian(delta_m_draws)) if delta_m_draws.size > 0 else np.nan
        self.delta_m_psf_err = float(np.nanstd(delta_m_draws)) if delta_m_draws.size > 0 else np.nan
        self.eta_psf = float(np.nanmedian(eta_psf_draws)) if eta_psf_draws.size > 0 else np.nan
        self.eta_psf_err = float(np.nanstd(eta_psf_draws)) if eta_psf_draws.size > 0 else np.nan
        self.scale_psf = 10.0 ** (-0.4 * self.delta_m_psf) if np.isfinite(self.delta_m_psf) else np.nan
        def _optional_draw_summary(key):
            """Return median/std for an optional predictive diagnostic."""
            if key not in pred_out:
                return np.nan, np.nan
            draws = np.asarray(pred_out[key], dtype=float)
            finite = np.isfinite(draws)
            if draws.size == 0 or not np.any(finite):
                return np.nan, np.nan
            return float(np.nanmedian(draws)), float(np.nanstd(draws))

        self.host_redshift_prior_weight, self.host_redshift_prior_weight_err = _optional_draw_summary('host_redshift_prior_weight')
        self.host_redshift_prior_loc_eff, self.host_redshift_prior_loc_eff_err = _optional_draw_summary('host_redshift_prior_loc_eff')
        self.host_redshift_prior_scale_eff, self.host_redshift_prior_scale_eff_err = _optional_draw_summary('host_redshift_prior_scale_eff')
        self.host_redshift_prior_df_eff, self.host_redshift_prior_df_eff_err = _optional_draw_summary('host_redshift_prior_df_eff')

        def _band(x):
            """Compute 16th/84th percentile uncertainty band across samples."""
            a = np.asarray(x)
            return np.percentile(a, 16, axis=0), np.percentile(a, 84, axis=0)

        cont_plus_lines = np.asarray(pred_out['continuum_model']) + np.asarray(pred_out['line_model'])
        fe_total = np.asarray(pred_out['f_fe_mgii_model']) + np.asarray(pred_out['f_fe_balmer_model'])
        self.pred_bands = {
            'total_model': _band(pred_out['model']),
            'host': _band(pred_out['gal_model']),
            'PL': _band(pred_out['f_pl_model']),
            'FeII': _band(fe_total),
            'Balmer_cont': _band(pred_out['f_bc_model']),
            'continuum': _band(pred_out['continuum_model']),
            'lines': _band(pred_out['line_model']),
            'conti_plus_lines': _band(cont_plus_lines),
        }
        if intrinsic_pl_draws is not None and intrinsic_pl_draws.shape[1] == len(self.wave):
            self.pred_bands['PL_intrinsic'] = _band(intrinsic_pl_draws)
        for name, draws in self._pred_custom_draws.items():
            self.pred_bands[name] = _band(draws)
        for name, draws in self._pred_custom_line_draws.items():
            self.pred_bands[name] = _band(draws)
        self.pred_bands_psf = {}
        if 'psf_model' in pred_out:
            self.pred_bands_psf['total_model'] = _band(pred_out['psf_model'])
        if 'gal_model_psf' in pred_out:
            self.pred_bands_psf['host'] = _band(pred_out['gal_model_psf'])
        if 'agn_model_psf' in pred_out:
            self.pred_bands_psf['PL'] = _band(pred_out['agn_model_psf'])
        if 'line_model_psf' in pred_out:
            self.pred_bands_psf['lines'] = _band(pred_out['line_model_psf'])
        if 'line_model_broad_psf' in pred_out:
            self.pred_bands_psf['line_broad'] = _band(pred_out['line_model_broad_psf'])
        if 'line_model_narrow_psf' in pred_out:
            self.pred_bands_psf['line_narrow'] = _band(pred_out['line_model_narrow_psf'])
        if bool(getattr(self, '_fit_fit_bal', False)):
            bi, bi_err = self.balnicity_index()
            self.bi = float(bi)
            self.bi_err = float(bi_err) if np.isfinite(bi_err) else np.nan
        if self.verbose:
            print("max data        :", np.nanmax(self.flux))
            print("max total model :", np.nanmax(self.model_total))
            print("max PL          :", np.nanmax(self.f_pl_model))
            print("max host        :", np.nanmax(self.host))
            print("max FeII UV     :", np.nanmax(self.f_fe_mgii_model))
            print("max FeII opt    :", np.nanmax(self.f_fe_balmer_model))
            print("max Balmer cont :", np.nanmax(self.f_bc_model))
            print("max lines       :", np.nanmax(self.f_line_model))
            for name, model in self.custom_components.items():
                print(f"max {name:<11}:", np.nanmax(model))
            for name, model in self.custom_line_components.items():
                print(f"max {name:<11}:", np.nanmax(model))

        if decompose_host and 'gal_v_kms' in samples and 'gal_sigma_kms' in samples:
            gal_v = float(np.median(np.asarray(samples['gal_v_kms'])))
            gal_v_err = float(np.std(np.asarray(samples['gal_v_kms'])))
            gal_sig = float(np.median(np.asarray(samples['gal_sigma_kms'])))
            gal_sig_err = float(np.std(np.asarray(samples['gal_sigma_kms'])))
        else:
            gal_v, gal_v_err, gal_sig, gal_sig_err = 0.0, 0.0, 0.0, 0.0

        ages = np.array([m['tage_gyr'] for m in fsps_grid.template_meta], dtype=float)
        mets = np.array([m['logzsol'] for m in fsps_grid.template_meta], dtype=float)
        wsum = np.sum(self.fsps_weights_median)
        age_weighted = float(np.sum(self.fsps_weights_median * ages) / wsum) if wsum > 0 else -1.0
        metal_weighted = float(np.sum(self.fsps_weights_median * mets) / wsum) if wsum > 0 else -99.0

        cont_waves = np.asarray(
            _continuum_output_waves_from_prior_config(
                getattr(self, "_fit_prior_config", None)
            ),
            dtype=float,
        )
        self.L_conti_wave = cont_waves
        pivot_wave = float(np.asarray(_spectrum_center_pivot(self.wave), dtype=float))

        frac_host_vals = []
        frac_host_psf_vals = []
        frac_bc_vals = []
        log_lambda_llambda_vals = []
        log_lambda_llambda_errs = []
        frac_host_names = []
        frac_host_psf_names = []
        frac_bc_names = []
        log_lambda_llambda_names = []
        log_lambda_llambda_err_names = []
        for w0 in cont_waves:
            wave_label = _format_wave_label(w0)
            frac_host = self._host_fraction_at_wave(w0)
            frac_host_psf = self._host_fraction_psf_at_wave(w0)
            frac_bc = self._bc_fraction_at_wave(w0)
            lum_key = f'log_lambda_Llambda_{wave_label}_agn'
            lum_draws = (
                np.asarray(pred_out[lum_key], dtype=float)
                if lum_key in pred_out
                else np.array([np.nan], dtype=float)
            )
            log_lambda_llambda = float(np.nanmedian(lum_draws)) if lum_draws.size > 0 else np.nan
            log_lambda_llambda_err = float(np.nanstd(lum_draws)) if lum_draws.size > 0 else np.nan
            setattr(self, f'frac_host_{wave_label}', frac_host)
            setattr(self, f'frac_host_psf_{wave_label}', frac_host_psf)
            setattr(self, f'frac_bc_{wave_label}', frac_bc)
            setattr(self, lum_key, log_lambda_llambda)
            setattr(self, f'{lum_key}_err', log_lambda_llambda_err)
            frac_host_vals.append(frac_host)
            frac_host_psf_vals.append(frac_host_psf)
            frac_bc_vals.append(frac_bc)
            log_lambda_llambda_vals.append(log_lambda_llambda)
            log_lambda_llambda_errs.append(log_lambda_llambda_err)
            frac_host_names.append(f'frac_host_{wave_label}')
            frac_host_psf_names.append(f'frac_host_psf_{wave_label}')
            frac_bc_names.append(f'frac_bc_{wave_label}')
            log_lambda_llambda_names.append(lum_key)
            log_lambda_llambda_err_names.append(f'{lum_key}_err')

        # Preserve the legacy fixed-wavelength attributes for downstream compatibility.
        self.pivot_wave = pivot_wave
        self.frac_host_pivot = self._host_fraction_at_wave(pivot_wave)
        self.frac_host_psf_pivot = self._host_fraction_psf_at_wave(pivot_wave)
        self.frac_bc_pivot = self._bc_fraction_at_wave(pivot_wave)
        self.frac_host_4200 = self._host_fraction_at_wave(4200.0)
        self.frac_host_5100 = self._host_fraction_at_wave(5100.0)
        self.frac_host_2500 = self._host_fraction_at_wave(2500.0)
        self.frac_host_psf_4200 = self._host_fraction_psf_at_wave(4200.0)
        self.frac_host_psf_5100 = self._host_fraction_psf_at_wave(5100.0)
        self.frac_host_psf_2500 = self._host_fraction_psf_at_wave(2500.0)
        self.frac_bc_2500 = self._bc_fraction_at_wave(2500.0)

        n_samp = int(np.asarray(next(iter(samples.values()))).shape[0]) if len(samples) > 0 else 1
        if 'PL_norm' in samples:
            pl_norm_samp = np.asarray(samples['PL_norm'])
        else:
            pl_norm_samp = np.full((n_samp,), np.nan)
        if 'PL_slope' in samples:
            pl_slope_med = float(np.nanmedian(np.asarray(samples['PL_slope'])))
            pl_slope_err = float(np.nanstd(np.asarray(samples['PL_slope'])))
        else:
            pl_slope_med = np.nan
            pl_slope_err = np.nan
        if 'reddening_a2500' in samples:
            reddening_a2500_med = float(np.nanmedian(np.asarray(samples['reddening_a2500'])))
            reddening_a2500_err = float(np.nanstd(np.asarray(samples['reddening_a2500'])))
        else:
            reddening_a2500_med = np.nan
            reddening_a2500_err = np.nan
        conti_entries = [
            ('ra', self.ra, 'float'),
            ('dec', self.dec, 'float'),
            ('filename', str(self.filename), 'str'),
            ('redshift', self.z, 'float'),
            ('SN_ratio_conti', self.SN_ratio_conti, 'float'),
            ('PL_norm', float(np.nanmedian(pl_norm_samp)), 'float'),
            ('PL_norm_err', float(np.nanstd(pl_norm_samp)), 'float'),
            ('PL_slope', pl_slope_med, 'float'),
            ('PL_slope_err', pl_slope_err, 'float'),
            ('reddening_a2500', reddening_a2500_med, 'float'),
            ('reddening_a2500_err', reddening_a2500_err, 'float'),
            ('pivot_wave', self.pivot_wave, 'float'),
            ('frac_host_pivot', self.frac_host_pivot, 'float'),
            ('frac_host_psf_pivot', self.frac_host_psf_pivot, 'float'),
            ('frac_bc_pivot', self.frac_bc_pivot, 'float'),
            ('sigma', gal_sig, 'float'),
            ('sigma_err', gal_sig_err, 'float'),
            ('v_off', gal_v, 'float'),
            ('v_off_err', gal_v_err, 'float'),
        ]
        conti_entries += [(name, value, 'float') for name, value in zip(frac_host_names, frac_host_vals)]
        conti_entries += [(name, value, 'float') for name, value in zip(frac_host_psf_names, frac_host_psf_vals)]
        conti_entries += [(name, value, 'float') for name, value in zip(frac_bc_names, frac_bc_vals)]
        conti_entries += [
            (name, value, 'float')
            for name, value in zip(log_lambda_llambda_names, log_lambda_llambda_vals)
        ]
        conti_entries += [
            (name, value, 'float')
            for name, value in zip(log_lambda_llambda_err_names, log_lambda_llambda_errs)
        ]
        conti_entries += [
            ('fsps_age_weighted_gyr', age_weighted, 'float'),
            ('fsps_logzsol_weighted', metal_weighted, 'float'),
            ('host_redshift_prior_weight', self.host_redshift_prior_weight, 'float'),
            ('host_redshift_prior_weight_err', self.host_redshift_prior_weight_err, 'float'),
            ('host_redshift_prior_loc_eff', self.host_redshift_prior_loc_eff, 'float'),
            ('host_redshift_prior_loc_eff_err', self.host_redshift_prior_loc_eff_err, 'float'),
            ('host_redshift_prior_scale_eff', self.host_redshift_prior_scale_eff, 'float'),
            ('host_redshift_prior_scale_eff_err', self.host_redshift_prior_scale_eff_err, 'float'),
            ('host_redshift_prior_df_eff', self.host_redshift_prior_df_eff, 'float'),
            ('host_redshift_prior_df_eff_err', self.host_redshift_prior_df_eff_err, 'float'),
            ('delta_m_psf', self.delta_m_psf, 'float'),
            ('delta_m_psf_err', self.delta_m_psf_err, 'float'),
            ('eta_psf', self.eta_psf, 'float'),
            ('eta_psf_err', self.eta_psf_err, 'float'),
        ]

        self.conti_result, self.conti_result_type, self.conti_result_name = self._build_result_arrays(conti_entries)

        if use_lines and tied_line_meta['n_lines'] > 0:
            amp_comp = np.asarray(pred_out['line_amp_per_component'])
            mu_comp = np.asarray(pred_out['line_mu_per_component'])
            sig_comp = np.asarray(pred_out['line_sig_per_component'])

            amp_med = np.median(amp_comp, axis=0)
            amp_err = np.std(amp_comp, axis=0)
            mu_med = np.median(mu_comp, axis=0)
            mu_err = np.std(mu_comp, axis=0)
            sig_med = np.median(sig_comp, axis=0)
            sig_err = np.std(sig_comp, axis=0)

            vals, names, types = [], [], []
            for i, nm in enumerate(tied_line_meta['names']):
                vals.extend([amp_med[i], amp_err[i], mu_med[i], mu_err[i], sig_med[i], sig_err[i]])
                names.extend([f'{nm}_scale', f'{nm}_scale_err', f'{nm}_centerwave', f'{nm}_centerwave_err', f'{nm}_sigma', f'{nm}_sigma_err'])
                types.extend(['float'] * 6)

            self.line_result = np.array(vals, dtype=object)
            self.line_result_type = np.array(types, dtype=object)
            self.line_result_name = np.array(names, dtype=object)
            self.gauss_result = self.line_result
            self.gauss_result_name = self.line_result_name
            self.line_component_amp_median = amp_med
            self.line_component_mu_median = mu_med
            self.line_component_sig_median = sig_med
        else:
            self.line_result = np.array([])
            self.line_result_type = np.array([])
            self.line_result_name = np.array([])
            self.gauss_result = np.array([])
            self.gauss_result_name = np.array([])
            self.line_component_amp_median = np.array([])
            self.line_component_mu_median = np.array([])
            self.line_component_sig_median = np.array([])
        self._posterior_hydrated = True

    def _wave_trim(self, lam, flux, err, z):
        """Apply rest-frame wavelength range trimming.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Redshift used for rest-frame conversion.
        """
        ind_trim = np.where((lam / (1 + z) > self.wave_range[0]) & (lam / (1 + z) < self.wave_range[1]), True, False)
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError('No enough pixels in the input wave_range!')
        return self.lam, self.flux, self.err

    def _wave_msk(self, lam, flux, err, z):
        """Mask user-provided rest-frame wavelength intervals.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Redshift used for rest-frame conversion.
        """
        for msk in range(len(self.wave_mask)):
            ind_not_mask = ~np.where((lam / (1 + z) > self.wave_mask[msk, 0]) & (lam / (1 + z) < self.wave_mask[msk, 1]), True, False)
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err

    def _mask_lya_forest(self, lam, flux, err, z, lya_rest=1215.67):
        """Mask observed pixels blueward of rest-frame Ly-alpha.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Redshift used for rest-frame conversion.
        lya_rest : float, optional
            Rest-frame Ly-alpha cutoff in Angstrom.
        """
        keep = (lam / (1 + z)) >= float(lya_rest)
        self.lam, self.flux, self.err = lam[keep], flux[keep], err[keep]
        if len(self.lam) < 10:
            raise RuntimeError('Not enough pixels after Ly-alpha forest masking.')
        return self.lam, self.flux, self.err

    def _de_redden(self, lam, flux, err, ra, dec):
        """Correct observed flux/error for Galactic extinction using dustmaps.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        ra, dec : float
            Sky coordinates in degrees.
        """
        sfd_query = _get_sfd_query()
        coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame='icrs')
        ebv = float(np.asarray(sfd_query(coord)))
        self.ebv_mw = ebv
        zero_flux = np.where(flux == 0, True, False)
        flux[zero_flux] = 1e-10
        flux_unred = unred(lam, flux, ebv)
        err_unred = err * flux_unred / flux
        flux_unred[zero_flux] = 0
        self.flux = flux_unred
        self.err = err_unred
        return self.flux

    @staticmethod
    def _validate_deredden_coordinates(ra, dec):
        """Validate sky coordinates before Galactic dereddening."""
        ra_f = float(ra)
        dec_f = float(dec)
        invalid_placeholder = (ra_f == -999.0 and dec_f == -999.0)
        invalid_range = (not np.isfinite(ra_f)) or (not np.isfinite(dec_f)) or (dec_f < -90.0) or (dec_f > 90.0)
        if invalid_placeholder or invalid_range:
            raise ValueError(
                "Galactic dereddening requires valid sky coordinates: "
                f"received ra={ra_f}, dec={dec_f}. "
                "Pass real source coordinates in `FitConfig.observation` or set "
                "`FitConfig.observation.apply_mw_deredden=False` for synthetic "
                "data or spectra without sky positions."
            )
        return ra_f, dec_f

    def _rest_frame(self, lam, flux, err, z):
        """Convert observed-frame spectra to rest-frame convention.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Source redshift.
        """
        self.wave = lam / (1 + z)
        self.flux = flux * (1 + z)
        self.err = err * (1 + z)
        return self.wave, self.flux, self.err

    def _orignial_spec(self, wave, flux, err):
        """Cache the pre-modeling spectrum for plotting/debugging.

        Parameters
        ----------
        wave, flux, err : ndarray
            Rest-frame wavelength, flux, and uncertainty arrays.
        """
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _calculate_sn(self, wave, flux, alter=True):
        """Estimate continuum S/N from standard windows or robust fallback.

        Parameters
        ----------
        wave, flux : ndarray
            Rest-frame wavelength and flux arrays.
        alter : bool, optional
            If True and standard windows are unavailable, use robust fallback.
        """
        ind5100 = np.where((wave > 5080) & (wave < 5130), True, False)
        ind3000 = np.where((wave > 3000) & (wave < 3050), True, False)
        ind1350 = np.where((wave > 1325) & (wave < 1375), True, False)
        if np.all(np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) < 10):
            if alter is False:
                self.SN_ratio_conti = -1.
                return self.SN_ratio_conti
            input_data = np.array(flux)
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            if n > 4:
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                self.SN_ratio_conti = float(signal / noise)
            else:
                self.SN_ratio_conti = -1.
        else:
            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(), flux[ind1350].mean() / flux[ind1350].std()])
            tmp_SN = tmp_SN[np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) > 10]
            self.SN_ratio_conti = np.nanmean(tmp_SN) if not np.all(np.isnan(tmp_SN)) else -1.
        return self.SN_ratio_conti

    def _host_fraction_at_wave(self, w0):
        """Return host/continuum flux fraction at wavelength ``w0``.

        Parameters
        ----------
        w0 : float
            Rest-frame wavelength in Angstrom.
        """
        return self._component_fraction_at_wave(self.host, w0)

    def _host_fraction_psf_at_wave(self, w0):
        """Return PSF-space host fraction at wavelength ``w0``."""
        qso_psf = np.asarray(getattr(self, 'qso_psf', []), dtype=float)
        host_psf = np.asarray(getattr(self, 'host_psf', []), dtype=float)
        if qso_psf.size != len(getattr(self, 'wave', [])) or host_psf.size != len(getattr(self, 'wave', [])):
            return -1.0
        return self._component_fraction_at_wave(host_psf, w0, reference=qso_psf + host_psf)

    def _bc_fraction_at_wave(self, w0):
        """Return Balmer-continuum/continuum flux fraction at wavelength ``w0``.

        Parameters
        ----------
        w0 : float
            Rest-frame wavelength in Angstrom.
        """
        return self._component_fraction_at_wave(self.f_bc_model, w0)

    def _component_fraction_at_wave(self, component, w0, reference=None):
        """Return component fraction relative to fitted continuum at ``w0``.

        Parameters
        ----------
        component : ndarray
            Component flux array evaluated on ``self.wave``.
        w0 : float
            Rest-frame wavelength in Angstrom.
        reference : ndarray or None, optional
            Reference flux array. If ``None``, uses ``self.f_conti_model``.
        """
        if len(self.wave) == 0:
            return -1.
        comp = np.interp(w0, self.wave, component, left=np.nan, right=np.nan)
        ref_arr = self.f_conti_model if reference is None else np.asarray(reference, dtype=float)
        if len(ref_arr) != len(self.wave):
            return -1.
        total = np.interp(w0, self.wave, ref_arr, left=np.nan, right=np.nan)
        if not np.isfinite(comp) or not np.isfinite(total) or total == 0:
            return -1.
        return float(comp / total)

    def reconstruct_posterior_spectrum(
        self,
        wave_out=None,
        wave_min=2500.0,
        wave_max=None,
        n_draws=None,
        return_components=True,
        _state: _PosteriorState | None = None,
    ):
        """Rebuild posterior component draws on a requested rest-frame grid.

        Parameters
        ----------
        wave_out : array-like or None, optional
            Explicit rest-frame wavelength grid. If ``None``, build a grid from
            ``min(wave_min, self.wave.min())`` to ``wave_max or self.wave.max()``
            using the median native wavelength spacing.
        wave_min, wave_max : float or None, optional
            Bounds for the auto-generated grid when ``wave_out`` is ``None``.
        n_draws : int or None, optional
            If provided, use at most the first ``n_draws`` posterior samples.
        return_components : bool, optional
            If True, include per-component draws and medians in the return value.
            This includes any fitted custom components.
        """
        state = self._ensure_posterior_state() if _state is None else _state
        if state.samples is None:
            raise RuntimeError("No posterior samples available. Run fit() first.")
        has_age_grid = hasattr(self, '_fit_fsps_age_grid')
        has_logz_grid = hasattr(self, '_fit_fsps_logzsol_grid')
        has_fsps_grid = hasattr(self, 'fsps_grid')
        if not (has_age_grid and has_logz_grid) and not has_fsps_grid:
            raise RuntimeError("No template-grid metadata available for reconstruction.")
        if not hasattr(self, 'wave') or len(self.wave) < 2:
            raise RuntimeError("No fitted rest-frame wavelength grid available.")

        wave_native = np.asarray(self.wave, dtype=float)
        dw = float(np.nanmedian(np.diff(wave_native)))
        if not np.isfinite(dw) or dw <= 0:
            raise RuntimeError("Unable to infer wavelength spacing for reconstruction.")

        if wave_out is None:
            wmin = min(float(wave_min), float(np.nanmin(wave_native)))
            wmax = float(np.nanmax(wave_native) if wave_max is None else wave_max)
            if wmin >= wmax:
                raise ValueError("Requested reconstruction grid has non-positive span.")
            dln = float(np.nanmedian(np.diff(np.log(np.asarray(wave_native, dtype=float)))))
            if not np.isfinite(dln) or dln <= 0:
                raise RuntimeError("Unable to infer logarithmic wavelength spacing for reconstruction.")
            ln_grid = np.arange(np.log(wmin), np.log(wmax) + 0.5 * dln, dln, dtype=float)
            wave_out = np.exp(ln_grid)
            wave_out[0] = wmin
        else:
            wave_out = np.asarray(wave_out, dtype=float)

        if wave_out.ndim != 1 or wave_out.size < 2 or not np.all(np.isfinite(wave_out)):
            raise ValueError("wave_out must be a finite 1D wavelength grid.")

        prior_config = getattr(self, '_fit_prior_config', None)
        if prior_config is None:
            prior_config = _materialize_prior_config(build_default_prior_config(np.asarray(self.flux, dtype=float)))
        else:
            prior_config = dict(prior_config)
        if prior_config.get("PL_pivot", None) is None:
            prior_config["PL_pivot"] = float(np.asarray(_spectrum_center_pivot(wave_native), dtype=float))
        if prior_config.get("poly_pivot", None) is None:
            prior_config["poly_pivot"] = float(np.asarray(_spectrum_center_pivot(wave_native), dtype=float))
        age_grid_gyr, logzsol_grid, dsps_ssp_fn = self._require_posterior_bundle_fsps_metadata(self.__dict__)
        expected_templates = int(len(age_grid_gyr) * len(logzsol_grid))
        self._validate_fsps_weights_shape(
            state.predictive,
            expected_templates=expected_templates,
            context="Posterior reconstruction",
        )
        return reconstruct_posterior_components(
            wave_out=wave_out,
            samples=state.samples,
            pred_out=state.predictive,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
            prior_config=prior_config,
            fit_poly=bool(getattr(self, '_fit_fit_poly', False)),
            fit_reddening=bool(getattr(self, '_fit_fit_reddening', False)),
            fit_poly_order=int(getattr(self, '_fit_fit_poly_order', 2)),
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            custom_components=getattr(self, '_fit_custom_components', ()),
            template_norms=getattr(self, '_fit_fsps_template_norms', None),
            n_draws=n_draws,
            return_components=return_components,
            decompose_host=bool(getattr(self, '_fit_decompose_host', True)),
        )

    def component_fraction_at_wave(self, component='host', wave0=2500.0, reference='continuum', reconstruct=False, n_draws=None):
        """Return component/reference flux fraction at a requested wavelength.

        Parameters
        ----------
        component, reference : str, optional
            Component names. Supported reconstructed names are ``host``, ``PL``,
            ``Fe_uv``, ``Fe_op``, ``Balmer_cont``, and ``continuum``.
            Any fitted custom component names are also accepted.
        wave0 : float, optional
            Rest-frame wavelength in Angstrom.
        reconstruct : bool, optional
            If True, rebuild posterior components on a grid that reaches ``wave0``.
            Returns ``(median, err)`` from the posterior draws.
        n_draws : int or None, optional
            Maximum number of posterior draws to use in the reconstruction.
        """
        if not reconstruct:
            component_map = {
                'host': getattr(self, 'host', None),
                'Balmer_cont': getattr(self, 'f_bc_model', None),
                'continuum': getattr(self, 'f_conti_model', None),
            }
            component_map.update(getattr(self, 'custom_components', {}))
            comp_arr = component_map.get(component)
            ref_arr = component_map.get(reference, getattr(self, 'f_conti_model', None))
            if comp_arr is None or ref_arr is None or len(self.wave) == 0:
                return -1.0, np.nan
            comp = np.interp(wave0, self.wave, comp_arr, left=np.nan, right=np.nan)
            ref = np.interp(wave0, self.wave, ref_arr, left=np.nan, right=np.nan)
            if not np.isfinite(comp) or not np.isfinite(ref) or ref == 0:
                return -1.0, np.nan
            return float(comp / ref), np.nan

        recon = self.reconstruct_posterior_spectrum(wave_min=min(float(wave0), float(np.nanmin(self.wave))), n_draws=n_draws)
        wave = recon['wave']
        idx = int(np.argmin(np.abs(wave - float(wave0))))
        if component not in recon['draws'] or reference not in recon['draws']:
            raise ValueError(f"Unknown reconstructed component/reference: {component}, {reference}")
        num = np.asarray(recon['draws'][component], dtype=float)[:, idx]
        den = np.asarray(recon['draws'][reference], dtype=float)[:, idx]
        frac = np.divide(num, den, out=np.full_like(num, np.nan), where=np.isfinite(den) & (den != 0))
        good = np.isfinite(frac)
        if not np.any(good):
            return np.nan, np.nan
        p16, p50, p84 = np.percentile(frac[good], [16.0, 50.0, 84.0])
        return float(p50), float(0.5 * (p84 - p16))

    @staticmethod
    def _balnicity_index_from_arrays(
        wave: np.ndarray,
        bal_sum: np.ndarray,
        reference: np.ndarray,
        line_center: float,
        vmin: float,
        vmax: float,
        min_width: float,
        depth_threshold: float,
    ) -> tuple[float, list[tuple[float, float]]]:
        """Compute a simple BI-like integral from a BAL model and reference model."""
        wave = np.asarray(wave, dtype=float)
        bal_sum = np.asarray(bal_sum, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if wave.ndim != 1 or bal_sum.shape != wave.shape or reference.shape != wave.shape or wave.size < 2:
            return 0.0, []

        finite = np.isfinite(wave) & np.isfinite(bal_sum) & np.isfinite(reference) & (reference > 0)
        if not np.any(finite):
            return 0.0, []

        vel = C_KMS * (float(line_center) / wave - 1.0)
        sel = finite & (vel >= float(vmin)) & (vel <= float(vmax))
        if np.count_nonzero(sel) < 2:
            return 0.0, []

        vel_sel = vel[sel]
        bal_sel = bal_sum[sel]
        ref_sel = reference[sel]
        order = np.argsort(vel_sel)
        vel_sel = vel_sel[order]
        bal_sel = bal_sel[order]
        ref_sel = ref_sel[order]

        flux_norm = 1.0 + bal_sel / ref_sel
        integrand = 1.0 - flux_norm / 0.9
        active = np.isfinite(integrand) & (integrand > 0.0) & ((-bal_sel / ref_sel) >= float(depth_threshold))
        if not np.any(active):
            return 0.0, []

        bi_total = 0.0
        troughs: list[tuple[float, float]] = []
        idx = np.flatnonzero(active)
        start = idx[0]
        prev = idx[0]
        for cur in idx[1:]:
            if cur != prev + 1:
                v0 = float(vel_sel[start])
                v1 = float(vel_sel[prev])
                if (v1 - v0) >= float(min_width):
                    bi_total += float(np.trapezoid(integrand[start:prev + 1], vel_sel[start:prev + 1]))
                    troughs.append((v0, v1))
                start = cur
            prev = cur
        v0 = float(vel_sel[start])
        v1 = float(vel_sel[prev])
        if (v1 - v0) >= float(min_width):
            bi_total += float(np.trapezoid(integrand[start:prev + 1], vel_sel[start:prev + 1]))
            troughs.append((v0, v1))
        return float(max(bi_total, 0.0)), troughs

    def balnicity_index(
        self,
        component_names=None,
        line_center: float = 1549.06,
        vmin: float = 3000.0,
        vmax: float = 25000.0,
        min_width: float = 2000.0,
        depth_threshold: float = 0.1,
        include_line_emission: bool = True,
        return_details: bool = False,
    ):
        """Return a simple BALnicity-style index from the summed fitted BAL model.

        The BAL model is defined as the sum of selected negative custom
        components, typically names beginning with ``bal_``. The reference model
        is the BAL-free AGN continuum, optionally plus the fitted emission-line
        model. The returned BI uses the standard-style integrand
        ``1 - f_norm / 0.9`` over contiguous troughs at least ``min_width`` wide
        and deeper than ``depth_threshold``.
        """
        self._ensure_hydrated_from_samples()
        if not hasattr(self, 'wave') or len(self.wave) == 0:
            raise RuntimeError("No fitted spectrum available. Run fit() first.")

        custom_models = getattr(self, 'custom_components', {})
        if component_names is None:
            selected_names = [name for name in custom_models if str(name).startswith('bal_')]
        elif isinstance(component_names, str):
            selected_names = [component_names]
        else:
            selected_names = [str(name) for name in component_names]
        selected_names = [name for name in selected_names if name in custom_models]

        if len(selected_names) == 0:
            result = {
                'bi': 0.0,
                'bi_err': np.nan,
                'component_names': [],
                'troughs_kms': [],
                'line_center': float(line_center),
                'vmin': float(vmin),
                'vmax': float(vmax),
                'min_width': float(min_width),
                'depth_threshold': float(depth_threshold),
            }
            return result if return_details else (0.0, np.nan)

        bal_sum = np.sum([np.asarray(custom_models[name], dtype=float) for name in selected_names], axis=0)
        qso_model = np.asarray(getattr(self, 'qso', np.zeros_like(self.wave)), dtype=float)
        line_model = np.asarray(getattr(self, 'f_line_model', np.zeros_like(self.wave)), dtype=float) if include_line_emission else np.zeros_like(self.wave, dtype=float)
        reference = qso_model - bal_sum + line_model

        bi_med, troughs = self._balnicity_index_from_arrays(
            wave=np.asarray(self.wave, dtype=float),
            bal_sum=bal_sum,
            reference=reference,
            line_center=float(line_center),
            vmin=float(vmin),
            vmax=float(vmax),
            min_width=float(min_width),
            depth_threshold=float(depth_threshold),
        )

        bi_err = np.nan
        if hasattr(self, 'pred_out') and self.pred_out is not None and hasattr(self, '_pred_custom_draws'):
            draw_list = [np.asarray(self._pred_custom_draws[name], dtype=float) for name in selected_names if name in self._pred_custom_draws]
            qso_draws = np.asarray(self.pred_out.get('agn_model', []), dtype=float)
            line_draws = np.asarray(self.pred_out.get('line_model', []), dtype=float) if include_line_emission else np.zeros_like(qso_draws, dtype=float)
            if len(draw_list) == len(selected_names) and qso_draws.ndim == 2 and qso_draws.shape[1] == len(self.wave):
                bal_draws = np.sum(draw_list, axis=0)
                ref_draws = qso_draws - bal_draws + line_draws
                bi_draws = []
                for i in range(qso_draws.shape[0]):
                    bi_i, _ = self._balnicity_index_from_arrays(
                        wave=np.asarray(self.wave, dtype=float),
                        bal_sum=bal_draws[i],
                        reference=ref_draws[i],
                        line_center=float(line_center),
                        vmin=float(vmin),
                        vmax=float(vmax),
                        min_width=float(min_width),
                        depth_threshold=float(depth_threshold),
                    )
                    bi_draws.append(bi_i)
                bi_draws = np.asarray(bi_draws, dtype=float)
                good = np.isfinite(bi_draws)
                if np.any(good):
                    p16, p50, p84 = np.percentile(bi_draws[good], [16.0, 50.0, 84.0])
                    bi_med = float(p50)
                    bi_err = float(0.5 * (p84 - p16))

        result = {
            'bi': float(bi_med),
            'bi_err': float(bi_err) if np.isfinite(bi_err) else np.nan,
            'component_names': selected_names,
            'troughs_kms': troughs,
            'line_center': float(line_center),
            'vmin': float(vmin),
            'vmax': float(vmax),
            'min_width': float(min_width),
            'depth_threshold': float(depth_threshold),
        }
        return result if return_details else (result['bi'], result['bi_err'])

    def _line_profile_from_params(
        self,
        line_key: str,
        amp: np.ndarray,
        mu: np.ndarray,
        sig: np.ndarray,
    ) -> np.ndarray:
        """Build a line profile from explicit Gaussian parameter arrays.

        Parameters
        ----------
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        amp, mu, sig : ndarray
            Gaussian amplitudes, centers (ln lambda), and widths.
        """
        if not hasattr(self, 'wave') or len(self.wave) == 0:
            return np.array([], dtype=float)
        if not hasattr(self, 'tied_line_meta'):
            return np.zeros_like(self.wave, dtype=float)

        names = np.asarray(self.tied_line_meta.get('names', []))
        amp = np.asarray(amp, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sig = np.asarray(sig, dtype=float)
        if names.size == 0 or amp.size == 0 or mu.size == 0 or sig.size == 0:
            return np.zeros_like(self.wave, dtype=float)

        keep = np.array([str(n).startswith(f'{line_key}_') for n in names], dtype=bool)
        if not np.any(keep):
            return np.zeros_like(self.wave, dtype=float)

        lnw = np.log(np.asarray(self.wave, dtype=float))
        prof = np.zeros_like(lnw)
        for a, m, s in zip(amp[keep], mu[keep], sig[keep]):
            if np.isfinite(a) and np.isfinite(m) and np.isfinite(s) and s > 0:
                prof += a * np.exp(-0.5 * ((lnw - m) / s) ** 2)
        return prof

    def line_profile_from_components(self, line_key: str) -> np.ndarray:
        """Build a line-only profile from posterior-median component profiles.

        Parameters
        ----------
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        """
        profiles = np.asarray(getattr(self, 'line_component_profiles', []), dtype=float)
        names = np.asarray(getattr(self, 'tied_line_meta', {}).get('names', []))
        if profiles.ndim == 2 and profiles.shape[1] == len(self.wave) and names.size == profiles.shape[0]:
            keep = np.array([str(n).startswith(f'{line_key}_') for n in names], dtype=bool)
            if np.any(keep):
                return np.sum(profiles[keep], axis=0)
            return np.zeros_like(self.wave, dtype=float)
        if not hasattr(self, 'line_component_amp_median'):
            return np.zeros_like(self.wave, dtype=float)
        return self._line_profile_from_params(
            line_key=line_key,
            amp=np.asarray(getattr(self, 'line_component_amp_median', []), dtype=float),
            mu=np.asarray(getattr(self, 'line_component_mu_median', []), dtype=float),
            sig=np.asarray(getattr(self, 'line_component_sig_median', []), dtype=float),
        )

    def line_profile_from_draw(self, draw_index: int, line_key: str) -> np.ndarray:
        """Build a line-only profile for one posterior draw index.

        Parameters
        ----------
        draw_index : int
            Posterior draw index.
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        """
        self._ensure_hydrated_from_samples()
        if not hasattr(self, 'pred_out') or self.pred_out is None:
            return np.zeros_like(self.wave, dtype=float)
        names = np.asarray(getattr(self, 'tied_line_meta', {}).get('names', []))
        if 'line_component_profiles' in self.pred_out:
            profile_draws = np.asarray(self.pred_out['line_component_profiles'], dtype=float)
            if profile_draws.ndim == 3 and names.size == profile_draws.shape[1]:
                idx = int(draw_index)
                if idx < 0 or idx >= profile_draws.shape[0]:
                    raise IndexError(f'draw_index {idx} is out of bounds for {profile_draws.shape[0]} posterior draws')
                keep = np.array([str(n).startswith(f'{line_key}_') for n in names], dtype=bool)
                if np.any(keep):
                    return np.sum(profile_draws[idx, keep], axis=0)
                return np.zeros_like(self.wave, dtype=float)
        if 'line_amp_per_component' not in self.pred_out:
            return np.zeros_like(self.wave, dtype=float)

        amp_draws = np.asarray(self.pred_out['line_amp_per_component'])
        mu_draws = np.asarray(self.pred_out['line_mu_per_component'])
        sig_draws = np.asarray(self.pred_out['line_sig_per_component'])
        if amp_draws.ndim != 2 or mu_draws.ndim != 2 or sig_draws.ndim != 2:
            return np.zeros_like(self.wave, dtype=float)

        idx = int(draw_index)
        if idx < 0 or idx >= amp_draws.shape[0]:
            raise IndexError(f'draw_index {idx} is out of bounds for {amp_draws.shape[0]} posterior draws')

        return self._line_profile_from_params(
            line_key=line_key,
            amp=amp_draws[idx],
            mu=mu_draws[idx],
            sig=sig_draws[idx],
        )

    def line_props(self, profile: np.ndarray, wave: np.ndarray | None = None) -> tuple[float, float]:
        """Return ``(fwhm_kms, integrated_area)`` from a line profile.

        Parameters
        ----------
        profile : ndarray
            Line profile values.
        wave : ndarray or None, optional
            Wavelength array. If ``None``, ``self.wave`` is used.
        """
        p = np.asarray(profile, dtype=float)
        w = np.asarray(self.wave if wave is None else wave, dtype=float)
        if p.size == 0 or w.size == 0 or p.size != w.size:
            return np.nan, np.nan
        if not np.any(np.isfinite(p)) or np.nanmax(p) <= 0:
            return np.nan, np.nan

        ipeak = int(np.nanargmax(p))
        peak_lam = w[ipeak]
        half = 0.5 * p[ipeak]
        idx = np.where(p >= half)[0]
        area = float(np.trapezoid(np.clip(p, 0.0, None), w))
        if idx.size < 2 or not np.isfinite(peak_lam) or peak_lam <= 0:
            return np.nan, area

        fwhm_a = w[idx[-1]] - w[idx[0]]
        fwhm_kms = C_KMS * fwhm_a / peak_lam
        return float(fwhm_kms), area

    def line_props_from_profile(self, wave: np.ndarray, profile: np.ndarray) -> tuple[float, float]:
        """Compatibility wrapper for :meth:`line_props`.

        Parameters
        ----------
        wave : ndarray
            Wavelength array.
        profile : ndarray
            Line profile values.
        """
        return self.line_props(profile=profile, wave=wave)

    def save_result(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type, line_result_name, save_fits_name):
        """Write continuum+line summary table to a pandas CSV file.

        Parameters
        ----------
        conti_result, line_result : ndarray
            Continuum and line result values.
        conti_result_type, line_result_type : ndarray
            Legacy dtype tags (stored but not enforced).
        conti_result_name, line_result_name : ndarray
            Column names for continuum and line outputs.
        save_fits_name : str
            Output basename for CSV.
        """
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])
        df = pd.DataFrame([self.all_result], columns=self.all_result_name)
        out_dir = self.output_path if self.output_path is not None else '.'
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, save_fits_name + '.csv')
        df.to_csv(out_file, index=False)
        print(f"Saved results table: {out_file}")
        return

    def _posterior_series(self, param_names=None, max_vector_elems=2):
        """Flatten posterior samples into labeled 1D series for diagnostics."""
        from .plotting import posterior_series

        return posterior_series(self, param_names=param_names, max_vector_elems=max_vector_elems)

    @staticmethod
    def _build_result_arrays(entries):
        """Convert ``(name, value, type)`` entries into legacy result arrays."""
        return (
            np.array([value for _, value, _ in entries], dtype=object),
            np.array([dtype for _, _, dtype in entries], dtype=object),
            np.array([name for name, _, _ in entries], dtype=object),
        )

    @staticmethod
    def _filter_half_width_angstrom(filt):
        """Return an approximate half-width for a photometric filter."""
        from .plotting import filter_half_width_angstrom

        return filter_half_width_angstrom(filt)

    def _plot_filter_metadata(self, bands):
        """Return plotting metadata arrays for the requested photometric bands."""
        from .plotting import plot_filter_metadata

        return plot_filter_metadata(self, bands)

    @staticmethod
    def _style_axis(ax, spine_lw=1.5):
        """Apply consistent axis styling."""
        from .plotting import style_axis

        return style_axis(ax, spine_lw=spine_lw)

    def _synthetic_photometry_for_plot(self, model_attr='model_total'):
        """Return rest-frame synthetic photometry points for plotting, if available."""
        from .plotting import synthetic_photometry_for_plot

        return synthetic_photometry_for_plot(self, model_attr=model_attr)

    def _observed_photometry_for_plot(self):
        """Return rest-frame observed PSF photometry points for plotting, if available."""
        from .plotting import observed_photometry_for_plot

        return observed_photometry_for_plot(self)

    def plot_trace(
        self,
        param_names=None,
        max_vector_elems=2,
        save_fig_path=None,
        save_fig_name=None,
        show_plot=False,
    ):
        """Plot posterior trace series for selected parameters."""
        from .plotting import plot_trace

        return plot_trace(
            self,
            param_names=param_names,
            max_vector_elems=max_vector_elems,
            save_fig_path=save_fig_path,
            save_fig_name=save_fig_name,
            show_plot=show_plot,
        )

    def plot_corner(
        self,
        param_names=None,
        max_vector_elems=2,
        bins=30,
        max_points=5000,
        save_fig_path=None,
        save_fig_name=None,
        show_plot=False,
    ):
        """Plot posterior projections with ``corner.corner``."""
        from .plotting import plot_corner

        return plot_corner(
            self,
            param_names=param_names,
            max_vector_elems=max_vector_elems,
            bins=bins,
            max_points=max_points,
            save_fig_path=save_fig_path,
            save_fig_name=save_fig_name,
            show_plot=show_plot,
        )

    def plot_mcmc_diagnostics(self, do_trace=True, do_corner=True,
                              param_names=None,
                              max_vector_elems=2,
                              corner_bins=30, corner_max_points=2000,
                              save_fig_path=None,
                              show_plot=False):
        """Plot trace and/or corner diagnostics in a single convenience call."""
        from .plotting import plot_mcmc_diagnostics

        return plot_mcmc_diagnostics(
            self,
            do_trace=do_trace,
            do_corner=do_corner,
            param_names=param_names,
            max_vector_elems=max_vector_elems,
            corner_bins=corner_bins,
            corner_max_points=corner_max_points,
            save_fig_path=save_fig_path,
            show_plot=show_plot,
        )

    def plot_spectrum(self, **kwargs):
        """Plot the fitted spectrum, model components, and residuals."""
        from .plotting import plot_spectrum

        return plot_spectrum(self, **kwargs)

    def plot_fig(self, save_fig_path=None, broad_fwhm=1200, plot_legend=True, ylims=None, plot_residual=True, show_title=True,
                 plot_1sigma=True, sigma_alpha=0.12, show_plot=True, plot_psf_space=False, plot_intrinsic_powerlaw=False):
        """Plot data, model components, line decomposition, and residuals."""
        from .plotting import plot_fig

        return plot_fig(
            self,
            save_fig_path=save_fig_path,
            broad_fwhm=broad_fwhm,
            plot_legend=plot_legend,
            ylims=ylims,
            plot_residual=plot_residual,
            show_title=show_title,
            plot_1sigma=plot_1sigma,
            sigma_alpha=sigma_alpha,
            show_plot=show_plot,
            plot_psf_space=plot_psf_space,
            plot_intrinsic_powerlaw=plot_intrinsic_powerlaw,
        )
