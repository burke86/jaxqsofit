from __future__ import annotations

import os
import glob
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path

import extinction
import h5py
import matplotlib
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from jaxsedfit.filters import load_filter_curves

import jax
import jax.numpy as jnp
import optax
import numpyro.distributions as dist
from numpyro.handlers import reparam
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
from .defaults import (
    append_optional_line_rows,
    build_default_bal_components,
    _build_default_prior_config,
)
from .model import (
    C_KMS,
    _continuum_output_waves_from_prior_config,
    _extract_line_table_from_prior_config,
    _balmer_static_terms_jax,
    _format_wave_label,
    _get_sfd_query,
    _direct_width_site,
    _line_amplitude_site,
    _line_indices,
    _line_meta_broad_mask,
    _line_meta_int,
    _nlr_width_site,
    _normalize_template_flux,
    _np_to_jnp,
    _ordered_width_site,
    _spectrum_center_pivot,
    FSPSTemplateGrid,
    build_fsps_template_grid,
    build_orthogonal_polynomial_basis_config,
    build_tied_line_meta_from_linelist,
    extend_loglam_grid,
    qso_fsps_joint_model,
    reconstruct_posterior_components,
    unred,
)
from .results import FitResult, _PosteriorState, median_mapping

_SDSS_PSF_BANDS = ("u", "g", "r", "i", "z")
_SDSS_FILTER_CACHE = None


def _materialize_prior_config(prior_config) -> dict:
    """Return a mutable flat prior mapping for low-level model code.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    """
    if prior_config is None:
        return {}
    if isinstance(prior_config, PriorConfig):
        return prior_config.to_mapping()
    if hasattr(prior_config, "to_mapping"):
        return dict(prior_config.to_mapping())
    return dict(prior_config)


def _line_row_is_broad(row) -> bool:
    """Return True when a line-table row represents a broad component.


    Parameters
    ----------
    row : object
        row value.
    """
    name = str(row.get("linename", "")).lower()
    return name.endswith("_br") or ("_br" in name)


def _filter_line_table_by_kind(line_table, *, use_broad_lines=True, use_narrow_lines=True):
    """Filter built-in line rows by broad/narrow kind.

    Parameters
    ----------
    line_table : object
        line_table value.
    use_broad_lines : object
        use_broad_lines value.
    use_narrow_lines : object
        use_narrow_lines value.
    """
    if line_table is None:
        return None
    rows = []
    for row in line_table:
        row_is_broad = _line_row_is_broad(row)
        if row_is_broad and not bool(use_broad_lines):
            continue
        if not row_is_broad and not bool(use_narrow_lines):
            continue
        rows.append(dict(row))
    return rows


def _filter_prior_line_table_by_kind(prior_config, *, use_broad_lines=True, use_narrow_lines=True):
    """Return a prior mapping whose ``line.table`` respects line-kind switches.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    use_broad_lines : object
        use_broad_lines value.
    use_narrow_lines : object
        use_narrow_lines value.
    """
    if prior_config is None:
        return None
    prior_config = dict(prior_config)
    line_cfg = prior_config.get("line", None)
    if not isinstance(line_cfg, dict) or "table" not in line_cfg:
        return prior_config
    line_cfg = dict(line_cfg)
    line_cfg["table"] = _filter_line_table_by_kind(
        line_cfg.get("table"),
        use_broad_lines=use_broad_lines,
        use_narrow_lines=use_narrow_lines,
    )
    prior_config["line"] = line_cfg
    return prior_config


def _filter_custom_line_components_by_kind(custom_line_components, *, use_broad_lines=True, use_narrow_lines=True):
    """Filter custom line components by their broad/narrow classification.

    Parameters
    ----------
    custom_line_components : object
        custom_line_components value.
    use_broad_lines : object
        use_broad_lines value.
    use_narrow_lines : object
        use_narrow_lines value.
    """
    components = normalize_custom_line_components(custom_line_components)
    out = []
    for comp in components:
        line_kind = str(getattr(comp, "line_kind", "narrow")).lower()
        if line_kind == "broad" and not bool(use_broad_lines):
            continue
        if line_kind != "broad" and not bool(use_narrow_lines):
            continue
        out.append(comp)
    return tuple(out)


def _numpyro_geometry_reparam_config(
    prior_config,
    *,
    fit_pl=True,
    fit_fe=True,
    fit_bc=True,
    fit_poly=False,
    fit_reddening=False,
    fit_poly_order=2,
    decompose_host=True,
):
    """Build additional NumPyro reparameterizers for NUTS.

    The broad/narrow tied-line model already uses explicit non-centered
    standard-normal sites in physical model code. Do not add a global
    ``LocScaleReparam`` layer here: it rewrites physical sample-site names to
    decentered names, so Optax MAP warm starts keyed by physical names can miss
    the actual NUTS latent sites and produce bad post-Optax fits.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    fit_pl : object
        fit_pl value.
    fit_fe : object
        fit_fe value.
    fit_bc : object
        fit_bc value.
    fit_poly : object
        fit_poly value.
    fit_reddening : object
        fit_reddening value.
    fit_poly_order : object
        fit_poly_order value.
    decompose_host : object
        decompose_host value.
    """
    return {}


def _line_complex_dense_mass_blocks(tied_line_meta, *, standardized_amplitudes):
    """Return dense blocks for line complexes and the width hierarchy.

    Complexes without ordered broad components retain local amplitude/centroid
    blocks.  Complexes with ordered broad components move as complete units
    into the shared width block together with the global broad width and
    unordered width offsets.  Keeping their amplitudes, centroids, and ordered
    widths together captures the correlations induced by overlapping Gaussian
    components.  Sites are assigned to exactly one block, as required by
    NumPyro.
    """
    blocks = []
    width_complexes = list(
        tied_line_meta.get("broad_width_order_complex_indices", [])
    )
    width_labels = list(
        tied_line_meta.get("broad_width_order_site_labels", [])
    )
    centroid_hierarchies = list(
        tied_line_meta.get("broad_centroid_hierarchy_groups", [])
    )
    ordered_owner_indices = {int(index) for index in width_complexes}
    ordered_complex_sites = {}
    for complex_group in tied_line_meta.get("amp_complex_groups", []):
        complex_index = int(complex_group["complex_index"])
        complex_label = str(
            complex_group.get("site_label", f"complex_{complex_index}")
        )
        sites = [
            _line_amplitude_site(
                complex_label, standardized=standardized_amplitudes
            )
        ]
        for hierarchy_index, hierarchy in enumerate(centroid_hierarchies):
            if int(hierarchy.get("complex_index", -1)) == complex_index:
                sites.extend(
                    [
                        f"line_broad_center_{hierarchy_index}_std",
                        f"line_broad_relative_offsets_{hierarchy_index}_std",
                    ]
                )
        if complex_index in ordered_owner_indices:
            ordered_complex_sites[complex_index] = sites
        else:
            blocks.append(tuple(sites))

    unordered_ids = _line_meta_int(
        tied_line_meta,
        "unordered_width_group_ids",
        default=[],
    )
    width_sites = []
    if unordered_ids.size:
        n_w = int(tied_line_meta.get("n_wgroups", 0))
        wgroup = _line_meta_int(tied_line_meta, "wgroup", default=[])
        broad_mask = _line_meta_broad_mask(tied_line_meta)
        if n_w > 0 and wgroup.size == broad_mask.size:
            wgroup_is_broad = np.asarray(
                [np.any(broad_mask[wgroup == gid] > 0.0) for gid in range(n_w)],
                dtype=bool,
            )
            if np.any(wgroup_is_broad[unordered_ids]):
                width_sites.append("line_log_broad_fwhm_std")
        width_sites.append("line_log_fwhm_delta_group_std")
    added_ordered_complexes = set()
    for order_index, owner_index in enumerate(width_complexes):
        owner_index = int(owner_index)
        if owner_index not in added_ordered_complexes:
            width_sites.extend(ordered_complex_sites.get(owner_index, ()))
            added_ordered_complexes.add(owner_index)
        order_label = (
            str(width_labels[order_index])
            if order_index < len(width_labels)
            else str(order_index)
        )
        width_sites.append(_ordered_width_site(order_label, standardized=True))
    if width_sites:
        blocks.append(tuple(width_sites))
    return blocks


def _build_line_init_values(tied_line_meta, prior_config, *, use_lines=True):
    """Initialize all tied-line latent coordinates at their prior locations."""
    values = {}
    if not use_lines or int(tied_line_meta.get("n_lines", 0)) <= 0:
        return values

    n_v = int(tied_line_meta.get("n_vgroups", 0))
    n_w = int(tied_line_meta.get("n_wgroups", 0))
    n_f = int(tied_line_meta.get("n_fgroups", 0))
    if n_v > 0:
        independent_vgroups = _line_meta_int(
            tied_line_meta,
            "independent_vgroup_ids",
            default=np.arange(n_v),
        )
        if independent_vgroups.size:
            values["line_dmu_independent_group_std"] = np.zeros(
                independent_vgroups.size, dtype=float
            )
        nlr_families = tied_line_meta.get("nlr_vgroup_families", {})
        family_sites = {
            "low": "line_nlr_center_std",
            "high": "line_high_ion_offset_std",
            "coronal": "line_coronal_offset_std",
        }
        for family, site in family_sites.items():
            if _line_meta_int(nlr_families, family, default=[]).size:
                values[site] = np.array(0.0)
        for hierarchy_index, hierarchy in enumerate(
            tied_line_meta.get("broad_centroid_hierarchy_groups", [])
        ):
            values[f"line_broad_center_{hierarchy_index}_std"] = np.array(0.0)
            values[f"line_broad_relative_offsets_{hierarchy_index}_std"] = np.zeros(
                len(hierarchy["component_groups"]) - 1, dtype=float
            )

    if n_w > 0:
        wgroup = _line_meta_int(tied_line_meta, "wgroup")
        broad_mask = _line_meta_broad_mask(tied_line_meta)
        wgroup_is_broad = np.asarray(
            [np.any(broad_mask[wgroup == gid] > 0.0) for gid in range(n_w)],
            dtype=bool,
        )
        unordered_ids = _line_meta_int(
            tied_line_meta,
            "unordered_width_group_ids",
            default=np.arange(n_w),
        )
        if unordered_ids.size and np.any(wgroup_is_broad[unordered_ids]):
            values["line_log_broad_fwhm_std"] = np.array(0.0)
        if unordered_ids.size and np.any(~wgroup_is_broad[unordered_ids]):
            values["line_log_narrow_fwhm_std"] = np.array(0.0)
        if unordered_ids.size:
            values["line_log_fwhm_delta_group_std"] = np.zeros(
                unordered_ids.size, dtype=float
            )
        for direct_group in tied_line_meta.get("direct_width_groups", []):
            values[
                _direct_width_site(
                    str(direct_group["site_label"]), standardized=True
                )
            ] = np.array(0.0)
        for family, group_ids in tied_line_meta.get(
            "nlr_wgroup_families", {}
        ).items():
            if _line_indices(group_ids).size:
                values[_nlr_width_site(family, standardized=True)] = np.array(0.0)
        order_labels = list(
            tied_line_meta.get("broad_width_order_site_labels", [])
        )
        for order_index, group_ids in enumerate(
            tied_line_meta.get("broad_width_order_groups", [])
        ):
            order_label = (
                str(order_labels[order_index])
                if order_index < len(order_labels)
                else str(order_index)
            )
            values[_ordered_width_site(order_label, standardized=True)] = np.zeros(
                len(group_ids), dtype=float
            )

    if n_f > 0:
        amp_init = np.asarray(
            tied_line_meta.get("amp_init_group", np.zeros(n_f)), dtype=float
        )
        standardized = bool(prior_config.get("standardize_active_priors", False))
        for complex_group in tied_line_meta.get(
            "amp_complex_groups",
            [{"complex_index": 0, "fgroup_ids": list(range(n_f))}],
        ):
            complex_index = int(complex_group["complex_index"])
            complex_label = str(
                complex_group.get("site_label", f"complex_{complex_index}")
            )
            group_ids = _line_meta_int(complex_group, "fgroup_ids")
            site = _line_amplitude_site(
                complex_label, standardized=standardized
            )
            values[site] = (
                np.zeros(group_ids.size, dtype=float)
                if standardized
                else amp_init[group_ids]
            )
    return values


def _get_sdss_filters():
    """Load SDSS filter curves once and return a band->response mapping."""
    global _SDSS_FILTER_CACHE
    if _SDSS_FILTER_CACHE is None:
        filters = load_filter_curves([f"{band}_sdss" for band in _SDSS_PSF_BANDS])
        _SDSS_FILTER_CACHE = {band: filt for band, filt in zip(_SDSS_PSF_BANDS, filters)}
    return _SDSS_FILTER_CACHE


def _filter_wave_to_angstrom_array(value):
    """Return a filter wavelength grid as a float ndarray in Angstrom.

    Parameters
    ----------
    value : object
        value value.
    """
    if hasattr(value, "to_value"):
        return np.asarray(value.to_value(u.AA), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _filter_wave_to_angstrom_scalar(value):
    """Return a scalar wavelength-like object as a float in Angstrom.

    Parameters
    ----------
    value : object
        value value.
    """
    if hasattr(value, "to_value"):
        return float(value.to_value(u.AA))
    return float(value)


def _ab_mag_to_fnu(mag):
    """Convert AB magnitude to flux density in cgs units.


    Parameters
    ----------
    mag : object
        mag value.
    """
    mag = np.asarray(mag, dtype=np.float64)
    return 10.0 ** (-0.4 * (mag + 48.60))


def _fnu_to_ab_mag(fnu):
    """Convert flux density in cgs units to AB magnitude.

    Parameters
    ----------
    fnu : object
        fnu value.
    """
    fnu = np.asarray(fnu, dtype=np.float64)
    return -2.5 * np.log10(np.clip(fnu, 1e-300, None)) - 48.60


def _mw_band_attenuation_factor(wave_obs, filt_trans, ebv, r_v=3.1):
    """Return the AB-weighted Galactic attenuation factor through a filter.

    Parameters
    ----------
    wave_obs : object
        wave_obs value.
    filt_trans : object
        filt_trans value.
    ebv : object
        ebv value.
    r_v : object
        r_v value.
    """
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
        """Initialize a config-first JAXQSOFit spectral fitter.


        Parameters
        ----------
        config : object
            config value.
        """
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
        self.mask_in = (
            np.ones_like(self.flux_in, dtype=bool)
            if spec.mask is None
            else np.asarray(spec.mask, dtype=bool)
        )
        if spec.errors is None:
            self.err_in = np.full_like(self.flux_in, 1e-6, dtype=np.float64)
        else:
            err_arr = np.asarray(spec.errors, dtype=np.float64)
            if err_arr.ndim == 0:
                self.err_in = np.full_like(self.flux_in, float(err_arr), dtype=np.float64)
            else:
                self.err_in = err_arr
        self.z = float(obs.redshift)
        self.resolving_power = spec.resolving_power
        self.apply_instrumental_resolution = bool(spec.apply_instrumental_resolution)
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
        """numpyro_samples helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().samples = value

    @property
    def pred_out(self):
        """Posterior predictive outputs mirrored from the internal posterior state."""
        return self._ensure_posterior_state().predictive

    @pred_out.setter
    def pred_out(self, value) -> None:
        """pred_out helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().predictive = value

    @property
    def pred_bands(self):
        """Posterior uncertainty bands mirrored from the internal posterior state."""
        return self._ensure_posterior_state().bands

    @pred_bands.setter
    def pred_bands(self, value) -> None:
        """pred_bands helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().bands = value

    @property
    def fig(self):
        """Main fitted-spectrum figure mirrored from the internal posterior state."""
        return self._ensure_posterior_state().figure

    @fig.setter
    def fig(self, value) -> None:
        """fig helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().figure = value

    @property
    def trace_fig(self):
        """Trace figure mirrored from the internal posterior state."""
        return self._ensure_posterior_state().trace_figure

    @trace_fig.setter
    def trace_fig(self, value) -> None:
        """trace_fig helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().trace_figure = value

    @property
    def corner_fig(self):
        """Corner figure mirrored from the internal posterior state."""
        return self._ensure_posterior_state().corner_figure

    @corner_fig.setter
    def corner_fig(self, value) -> None:
        """corner_fig helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().corner_figure = value

    @property
    def _loaded_posterior_path(self):
        """Loaded posterior bundle path mirrored from the internal posterior state."""
        return self._ensure_posterior_state().path

    @_loaded_posterior_path.setter
    def _loaded_posterior_path(self, value) -> None:
        """_loaded_posterior_path helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().path = None if value is None else Path(value)

    @property
    def _posterior_hydrated(self) -> bool:
        """Whether posterior-derived products have been reconstructed."""
        return bool(self._ensure_posterior_state().hydrated)

    @_posterior_hydrated.setter
    def _posterior_hydrated(self, value: bool) -> None:
        """_posterior_hydrated helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().hydrated = bool(value)

    @property
    def _resumed_from_samples(self) -> bool:
        """Whether this fitter was loaded from a posterior bundle."""
        return bool(self._ensure_posterior_state().resumed_from_samples)

    @_resumed_from_samples.setter
    def _resumed_from_samples(self, value: bool) -> None:
        """_resumed_from_samples helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_posterior_state().resumed_from_samples = bool(value)

    @classmethod
    def from_arrays(
        cls,
        *,
        lam,
        flux,
        err=None,
        mask=None,
        z=0.0,
        ra=None,
        dec=None,
        filename=None,
        output_path=None,
        resolving_power=None,
        apply_instrumental_resolution=False,
        psf_mags=None,
        psf_mag_errs=None,
        psf_bands=None,
    ):
        """Build a config-first fitter from raw arrays.

        Parameters
        ----------
        lam : array-like
            Observed-frame wavelength array in Angstrom.
        flux : array-like
            Observed spectral flux-density array.
        err : array-like or float, optional
            Flux-density uncertainty. If omitted, a small positive uncertainty
            is supplied by the downstream data preparation.
        mask : array-like of bool, optional
            Pixel keep-mask. ``True`` includes a pixel and ``False`` rejects it.
        z : float, optional
            Source redshift.
        ra, dec : float, optional
            Sky coordinates in degrees, used for Milky Way dereddening when
            enabled.
        filename : str, optional
            Object name and default output basename.
        output_path : str or pathlib.Path, optional
            Directory for saved figures and posterior bundles.
        resolving_power : float, optional
            Effective resolving power used to downweight oversampled spectral
            likelihoods.
        apply_instrumental_resolution : bool, optional
            If True, include the Gaussian instrumental LSF in the forward model.
            Requires a positive ``resolving_power``. Defaults to False.
        psf_mags, psf_mag_errs : array-like, optional
            PSF-aperture magnitudes and uncertainties used for spectral
            recalibration.
        psf_bands : sequence of str, optional
            Filter names corresponding to ``psf_mags``.
        """
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
                mask=mask,
                resolving_power=resolving_power,
                apply_instrumental_resolution=apply_instrumental_resolution,
            ),
            psf_photometry=psf,
            output=OutputConfig(output_path=output_path, save_name=filename),
        )
        return cls(cfg)

    @staticmethod
    def _resolve_filename(filename=None, ra=-999, dec=-999):
        """Resolve a filesystem-safe basename for outputs.

        Parameters
        ----------
        filename : object
            filename value.
        ra : object
            ra value.
        dec : object
            dec value.
        """
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
        """Return only active posterior sites needed for summaries and plots.

        Keeping inactive line-profile and PSF arrays out of ``Predictive`` lets
        XLA eliminate those output branches from the post-sampling graph.  This
        reduces compilation and transfer overhead without changing fitted
        samples or the public summaries for enabled components.

        Parameters
        ----------
        custom_components : object
            custom_components value.
        custom_line_components : object
            custom_line_components value.
        """
        use_lines = bool(getattr(self, "_fit_fit_lines", True))
        use_psf_phot = bool(
            getattr(self, "_fit_use_psf_phot", getattr(self, "use_psf_phot", False))
        )
        return_sites = [
            'PL_norm',
            'PL_slope',
            'frac_jitter',
            'add_jitter',
            'f_pl_model',
            'f_fe_mgii_model',
            'f_fe_balmer_model',
            'f_bc_model',
            'f_poly_model',
            'ebv',
            'reddening_a2500',
            'agn_model',
            'gal_model',
            'line_model',
            'continuum_model',
            'model',
            'fsps_weights',
            'gal_sigma_effective_kms',
            'spectral_likelihood_weight',
        ]
        if bool(getattr(self, "_fit_fit_fe", True)):
            return_sites.append('frac_fe_jitter')
        if use_lines:
            return_sites += [
                'line_model_broad',
                'line_model_narrow',
                'line_component_profiles',
                'line_amp_per_component',
                'line_amp_group',
                'line_mu_per_component',
                'line_sig_per_component',
                'line_sig_effective_per_component',
                'line_amp_effective_per_component',
            ]
        if use_psf_phot:
            return_sites += [
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

        Parameters
        ----------
        wave_obs : object
            wave_obs value.
        psf_mags : object
            psf_mags value.
        psf_mag_errs : object
            psf_mag_errs value.
        psf_bands : object
            psf_bands value.
        use_psf_phot : object
            use_psf_phot value.
        min_filter_coverage : object
            min_filter_coverage value.
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
        """Return the compressed on-disk path for a saved posterior bundle.

        Parameters
        ----------
        save_name : object
            save_name value.
        save_path : object
            save_path value.
        """
        out_name = self._normalize_posterior_bundle_name(
            f"{self.filename}_samples" if save_name is None else save_name
        )
        out_dir = self.output_path if save_path is None else save_path
        if out_dir is None:
            out_dir = '.'
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, out_name)

    def _intrinsic_powerlaw_draws(self, wave_out=None, apply_psf_scale=False):
        """Return posterior draws for the intrinsic AGN power law on ``wave_out``.

        Parameters
        ----------
        wave_out : object
            wave_out value.
        apply_psf_scale : object
            apply_psf_scale value.
        """
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
        """Normalize posterior bundle names to the enforced ``.h5`` suffix.


        Parameters
        ----------
        name : object
            name value.
        """
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
        """Return True when value is a matplotlib figure/axes object.


        Parameters
        ----------
        value : object
            value value.
        """
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
        """Return True when an attribute should be skipped during bundle save.

        Parameters
        ----------
        key : object
            key value.
        value : object
            value value.
        """
        if key in cls._bundle_excluded_keys():
            return True
        if key.startswith("_pred_"):
            return True
        if cls._is_matplotlib_state(value):
            return True
        return False

    @staticmethod
    def _serialize_for_hdf5(value):
        """Recursively convert model state into HDF5-serializable objects.

        Parameters
        ----------
        value : object
            value value.
        """
        from .config import _numpyro_distribution_to_mapping

        prior = _numpyro_distribution_to_mapping(value)
        if prior is not None:
            return JAXQSOFit._serialize_for_hdf5(prior)
        if isinstance(value, CustomComponentSpec):
            return JAXQSOFit._serialize_for_hdf5(value.to_state())
        if isinstance(value, CustomLineComponentSpec):
            return JAXQSOFit._serialize_for_hdf5(value.to_state())
        if hasattr(value, "to_mapping"):
            return JAXQSOFit._serialize_for_hdf5(value.to_mapping())
        if is_dataclass(value) and not isinstance(value, type):
            return JAXQSOFit._serialize_for_hdf5(asdict(value))
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
        """Rebuild custom serialized objects after reading HDF5 state.

        Parameters
        ----------
        value : object
            value value.
        """
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
        """Write one recursively serialized Python value into an HDF5 group.

        Parameters
        ----------
        parent : object
            parent value.
        name : object
            name value.
        value : object
            value value.
        """
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
        """Read one recursively serialized Python value from an HDF5 group.

        Parameters
        ----------
        parent : object
            parent value.
        name : object
            name value.
        """
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
            "mask_in",
            "resolving_power",
            "apply_instrumental_resolution",
            "z",
            "ra",
            "dec",
            "filename",
            "output_path",
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
            'broad_mask': np.array([], dtype=float),
            'broad_mask_jax': _np_to_jnp(np.array([], dtype=float)),
            'names': [],
            'compnames': [],
            'line_lambda': np.array([], dtype=float),
        }

    @staticmethod
    def _require_posterior_bundle_fsps_metadata(state):
        """Return required FSPS bundle metadata or raise on incomplete bundles.

        Parameters
        ----------
        state : object
            state value.
        """
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
        """Ensure hydrated or reconstructed FSPS weights match the expected basis width.

        Parameters
        ----------
        pred_out : object
            pred_out value.
        expected_templates : object
            expected_templates value.
        context : object
            context value.
        """
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
            prior_config = _materialize_prior_config(_build_default_prior_config(flux))
        else:
            prior_config = _materialize_prior_config(prior_config)
        custom_components = normalize_custom_components(getattr(self, "_fit_custom_components", ()))
        custom_line_components = normalize_custom_line_components(getattr(self, "_fit_custom_line_components", ()))
        prior_config = inject_default_custom_component_priors(prior_config, flux, custom_components)
        prior_config = inject_default_custom_line_component_priors(prior_config, flux, custom_line_components)
        conti_priors = prior_config.get("conti_priors", {})

        use_lines = bool(getattr(self, "_fit_fit_lines", True))
        line_table = _extract_line_table_from_prior_config(prior_config)
        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(
                line_table,
                wave,
                pool_narrow_centroids=bool(
                    prior_config.get("pool_narrow_centroids", True)
                ),
            )
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
            **self._static_component_cache_kwargs(),
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
        """Persist posterior samples plus minimal metadata for compact reloads.

        Parameters
        ----------
        save_name : object
            save_name value.
        save_path : object
            save_path value.
        _state : object
            _state value.
        """
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
        """Persist posterior samples and fit metadata to a compact bundle.

        Parameters
        ----------
        path : object
            path value.
        save_name : object
            save_name value.
        _state : object
            _state value.
        """
        return self.save_posterior_bundle(save_name=save_name, save_path=path, _state=_state)

    @staticmethod
    def _build_fsps_grid_for_fit(wave, age_grid_gyr, logzsol_grid, dsps_ssp_fn, decompose_host, z_qso=0.0, host_pad_kms=3000.0):
        """Build the host-template grid only when host decomposition is enabled.

        Parameters
        ----------
        wave : object
            wave value.
        age_grid_gyr : object
            age_grid_gyr value.
        logzsol_grid : object
            logzsol_grid value.
        dsps_ssp_fn : object
            dsps_ssp_fn value.
        decompose_host : object
            decompose_host value.
        z_qso : object
            z_qso value.
        host_pad_kms : object
            Velocity-space support added to each side of the host grid.
        """
        if decompose_host:
            host_wave = extend_loglam_grid(wave, pad_kms=host_pad_kms)
            return build_fsps_template_grid(
                wave_out=host_wave,
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
        """Load a compressed HDF5 posterior bundle and return a JAXQSOFit object.

        Parameters
        ----------
        filename : object
            filename value.
        output_path : object
            output_path value.
        save_name : object
            save_name value.
        plot_fig : object
            plot_fig value.
        plot_diagnostics : object
            plot_diagnostics value.
        kwargs_plot : object
            kwargs_plot value.
        diagnostics_kwargs : object
            diagnostics_kwargs value.
        """
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
            mask=state.get("mask_in"),
            resolving_power=state.get("resolving_power"),
            apply_instrumental_resolution=state.get("apply_instrumental_resolution", False),
            z=state.get("z", 0.0),
            ra=state.get("ra", -999),
            dec=state.get("dec", -999),
            filename=state.get("filename", resolved_name),
            output_path=output_path if output_path is not None else state.get("output_path"),
            psf_mags=state.get("psf_mags_raw", state.get("psf_mags")),
            psf_mag_errs=state.get("psf_mag_errs_raw", state.get("psf_mag_errs")),
            psf_bands=state.get("psf_bands"),
        )
        state.pop("wdisp", None)
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
        """Build a public result object from the current mirrored fit state.

        Parameters
        ----------
        method : object
            method value.
        path : object
            path value.
        figure : object
            figure value.
        """
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

    def _static_component_cache_kwargs(self) -> dict:
        """Return precomputed wavelength-grid component caches when available."""
        return {
            "fe_uv_flux_on_wave": getattr(self, "_fe_uv_flux_on_wave", None),
            "fe_op_flux_on_wave": getattr(self, "_fe_op_flux_on_wave", None),
            "balmer_bb_shape": getattr(self, "_balmer_bb_shape", None),
            "balmer_tau_shape": getattr(self, "_balmer_tau_shape", None),
            "balmer_below_edge": getattr(self, "_balmer_below_edge", None),
        }

    @classmethod
    def load_result(cls, *args, **kwargs) -> FitResult:
        """Load a posterior bundle and wrap it in a :class:`FitResult`.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        fitter = cls.load(*args, **kwargs)
        return fitter._make_result(
            method=getattr(fitter, "_fit_inference_method", "loaded"),
            path=getattr(fitter, "_loaded_posterior_path", None),
        )

    def fit(self, *, verbose=True, kwargs_plot=None, **kwargs):
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
        **kwargs
            Removed legacy fit keyword arguments. Passing any value here raises
            a configuration-first error message with the corresponding config
            field when one exists.

        Returns
        -------
        FitResult
            Result object exposing samples, medians, persistence, and plotting
            helpers while the fitter keeps mirrored posterior state.
        """
        if kwargs:
            legacy_targets = {
                "deredden": "config.observation.apply_mw_deredden",
                "fit_lines": "config.lines.enabled",
                "decompose_host": "config.host.enabled",
                "fit_pl": "config.continuum.fit_power_law",
                "fit_fe": "config.continuum.fit_feii",
                "fit_bc": "config.continuum.fit_balmer_continuum",
                "fit_poly": "config.continuum.fit_polynomial_tilt",
                "fit_reddening": "config.continuum.fit_reddening",
                "fit_poly_order": "config.continuum.polynomial_order",
                "save_result": "config.output.save_result",
                "plot_fig": "config.output.plot_fig",
                "save_fig": "config.output.save_fig",
                "output_path": "config.output.output_path",
                "fig_path": "config.output.output_path",
                "result_path": "config.output.output_path",
                "nuts_warmup": "config.inference.num_warmup",
                "nuts_samples": "config.inference.num_samples",
                "nuts_chains": "config.inference.num_chains",
                "target_accept_prob": "config.inference.target_accept_prob",
                "dense_mass": "config.inference.dense_mass",
                "max_tree_depth": "config.inference.max_tree_depth",
                "optax_steps": "config.inference.map_steps",
                "optax_lr": "config.inference.learning_rate",
                "fit_method": "config.inference.method",
                "method": "config.inference.method",
            }
            details = []
            for key in sorted(kwargs):
                target = legacy_targets.get(key)
                if target is None:
                    details.append(f"{key}: no public fit() keyword exists")
                else:
                    details.append(f"{key}: set q.{target} before q.fit()")
            joined = "; ".join(details)
            raise TypeError(
                "JAXQSOFit.fit() is configuration-first and does not accept "
                f"model/inference keyword arguments. {joined}."
            )

        cfg = self.config
        obs_cfg = cfg.observation
        prep_cfg = cfg.preprocessing
        cont_cfg = cfg.continuum
        host_cfg = cfg.host
        line_cfg = cfg.lines
        infer_cfg = cfg.inference
        out_cfg = cfg.output
        psf_cfg = cfg.psf_photometry
        bal_cfg = cfg.bal

        name = out_cfg.save_name
        deredden = bool(obs_cfg.apply_mw_deredden)
        wave_range = prep_cfg.wave_range
        wave_mask = prep_cfg.wave_mask
        mask_lya_forest = bool(prep_cfg.mask_lya_forest)
        fit_lines = bool(line_cfg.enabled)
        use_broad_lines = bool(line_cfg.use_broad_lines)
        use_narrow_lines = bool(line_cfg.use_narrow_lines)
        pool_narrow_centroids = bool(line_cfg.pool_narrow_centroids)
        include_elg_narrow_lines = bool(line_cfg.include_elg_narrow_lines)
        include_high_ionization_lines = bool(line_cfg.include_high_ionization_lines)
        decompose_host = bool(host_cfg.enabled)
        fit_pl = bool(cont_cfg.fit_power_law)
        fit_fe = bool(cont_cfg.fit_feii)
        fit_bc = bool(cont_cfg.fit_balmer_continuum)
        fit_bal = bool(bal_cfg.enabled)
        fit_poly = bool(cont_cfg.fit_polynomial_tilt)
        fit_reddening = bool(cont_cfg.fit_reddening)
        fit_poly_order = int(cont_cfg.polynomial_order)
        broadening_convolution = str(cont_cfg.broadening_convolution).lower()
        method = str(infer_cfg.method)
        random_seed = int(infer_cfg.random_seed)
        self._posterior_state = _PosteriorState(method=method)
        fsps_age_grid = host_cfg.age_grid_gyr
        fsps_logzsol_grid = host_cfg.logzsol_grid
        host_sfh_model = str(host_cfg.sfh_model)
        dsps_ssp_fn = host_cfg.dsps_ssp_fn
        nuts_warmup = int(infer_cfg.num_warmup)
        nuts_samples = int(infer_cfg.num_samples)
        nuts_chains = int(infer_cfg.num_chains)
        nuts_target_accept = float(infer_cfg.target_accept_prob)
        nuts_dense_mass = bool(infer_cfg.dense_mass)
        line_block_dense_mass = bool(infer_cfg.line_block_dense_mass)
        standardize_active_priors = bool(infer_cfg.standardize_active_priors)
        nuts_max_tree_depth = int(infer_cfg.max_tree_depth)
        optax_steps = int(infer_cfg.map_steps)
        optax_lr = float(infer_cfg.learning_rate)
        plot_init = bool(infer_cfg.plot_init or out_cfg.plot_init)
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
        self._fit_use_broad_lines = bool(use_broad_lines)
        self._fit_use_narrow_lines = bool(use_narrow_lines)
        self._fit_pool_narrow_centroids = bool(pool_narrow_centroids)
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
        self._fit_dsps_ssp_fn = str(dsps_ssp_fn)
        self._fit_use_psf_phot = bool(use_psf_phot)
        requested_custom_components = normalize_custom_components(custom_components)
        self._fit_custom_components = requested_custom_components
        self._fit_custom_line_components = _filter_custom_line_components_by_kind(
            custom_line_components,
            use_broad_lines=use_broad_lines,
            use_narrow_lines=use_narrow_lines,
        )

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

        pixel_keep = (
            self.mask_in
            & np.isfinite(self.lam_in)
            & (self.lam_in > 0.0)
            & np.isfinite(self.flux_in)
            & np.isfinite(self.err_in)
            & (self.err_in > 0.0)
        )
        self.err = self.err_in[pixel_keep]
        self.flux = self.flux_in[pixel_keep]
        self.lam = self.lam_in[pixel_keep]

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
        self._original_spec(self.wave, self.flux, self.err)
        self._fe_uv_flux_on_wave = np.interp(
            self.wave,
            self.fe_uv_wave,
            np.maximum(self.fe_uv_flux, 0.0),
            left=0.0,
            right=0.0,
        )
        self._fe_op_flux_on_wave = np.interp(
            self.wave,
            self.fe_op_wave,
            np.maximum(self.fe_op_flux, 0.0),
            left=0.0,
            right=0.0,
        )
        balmer_bb_shape, balmer_tau_shape, balmer_below_edge = _balmer_static_terms_jax(
            _np_to_jnp(self.wave),
            balmer_te=15000.0,
        )
        self._balmer_bb_shape = np.asarray(balmer_bb_shape, dtype=float)
        self._balmer_tau_shape = np.asarray(balmer_tau_shape, dtype=float)
        self._balmer_below_edge = np.asarray(balmer_below_edge, dtype=bool)
        resolving_power = self.resolving_power
        apply_instrumental_resolution = bool(self.apply_instrumental_resolution)
        if apply_instrumental_resolution and resolving_power is None:
            raise ValueError(
                "apply_instrumental_resolution=True requires SpectroscopyData.resolving_power."
            )
        if resolving_power is None:
            warnings.warn(
                "SpectroscopyData.resolving_power is None; jaxqsofit will treat spectral pixels as independent, "
                "so posterior uncertainties may be over-confident for oversampled spectra.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            resolving_power = float(resolving_power)
            if not np.isfinite(resolving_power) or resolving_power <= 0.0:
                raise ValueError("SpectroscopyData.resolving_power must be positive when specified.")

        bal_components = (
            build_default_bal_components(
                self.flux,
                tau_scale=float(bal_cfg.tau_scale),
                covering_loc=float(bal_cfg.covering_loc),
                covering_scale=float(bal_cfg.covering_scale),
                covering_high=float(bal_cfg.covering_high),
                fwhm_kms_loc=float(bal_cfg.fwhm_kms_loc),
                fwhm_kms_scale=float(bal_cfg.fwhm_kms_scale),
                fwhm_kms_low=float(bal_cfg.fwhm_kms_low),
                fwhm_kms_high=float(bal_cfg.fwhm_kms_high),
            )
            if bool(fit_bal)
            else ()
        )
        self._fit_custom_components = normalize_custom_components(
            tuple(requested_custom_components) + tuple(bal_components)
        )

        if prior_config_input is None:
            prior_config = _materialize_prior_config(_build_default_prior_config(self.flux))
        else:
            prior_config = _materialize_prior_config(prior_config)
        prior_config = append_optional_line_rows(
            prior_config,
            self.flux,
            include_elg_narrow_lines=include_elg_narrow_lines,
            include_high_ionization_lines=include_high_ionization_lines,
        )
        prior_config["convolution_method"] = broadening_convolution
        prior_config["standardize_active_priors"] = standardize_active_priors
        prior_config["line_block_dense_mass"] = line_block_dense_mass
        prior_config["pool_narrow_centroids"] = pool_narrow_centroids
        prior_config["z_qso"] = float(self.z)
        prior_config["host_sfh_model"] = str(host_sfh_model)
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
        prior_config = _filter_prior_line_table_by_kind(
            prior_config,
            use_broad_lines=use_broad_lines,
            use_narrow_lines=use_narrow_lines,
        )
        out_params = prior_config.get('out_params', {})
        self.L_conti_wave = np.asarray(out_params.get('cont_loc', []), dtype=float)

        pl_pivot = prior_config.get("PL_pivot", None)
        if pl_pivot is None:
            pl_pivot = _spectrum_center_pivot(self.wave)
        prior_config["PL_pivot"] = float(np.asarray(pl_pivot, dtype=float))
        poly_pivot = prior_config.get("poly_pivot", None)
        if poly_pivot is None:
            poly_pivot = _spectrum_center_pivot(self.wave)
        prior_config["poly_pivot"] = float(np.asarray(poly_pivot, dtype=float))
        if fit_poly:
            prior_config["poly_basis"] = build_orthogonal_polynomial_basis_config(
                self.wave,
                self.err,
                pivot=prior_config["poly_pivot"],
                order=fit_poly_order,
                include_reddening=fit_reddening,
                reddening_uv_ref=float(prior_config.get("reddening_uv_ref", 2500.0)),
                reddening_alpha=float(prior_config.get("reddening_alpha", 1.2)),
            )
        prior_config["resolving_power"] = resolving_power
        prior_config["apply_instrumental_resolution"] = apply_instrumental_resolution
        finalized_prior_config = _materialize_prior_config(prior_config)
        self._fit_prior_config = finalized_prior_config
        self._fit_host_sfh_model = str(
            finalized_prior_config.get("host_sfh_model", "flexible")
        )
        prior_config = finalized_prior_config
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
                dense_mass=nuts_dense_mass,
                max_tree_depth=nuts_max_tree_depth,
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
                random_seed=random_seed,
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
                random_seed=random_seed,
            )
        elif method == 'optax+nuts':
            self.run_fsps_optax_nuts_fit(
                optax_steps=optax_steps,
                optax_learning_rate=optax_lr,
                num_warmup=nuts_warmup,
                num_samples=nuts_samples,
                num_chains=nuts_chains,
                target_accept_prob=nuts_target_accept,
                dense_mass=nuts_dense_mass,
                max_tree_depth=nuts_max_tree_depth,
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
                random_seed=random_seed,
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
                             dense_mass=True,
                             max_tree_depth=8,
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
                             init_values=None, random_seed=0):
        """Fit the full model using NUTS MCMC and store posterior summaries.

        Parameters
        ----------
        num_warmup, num_samples : int, optional
            MCMC warmup and posterior sample counts.
        num_chains : int, optional
            Number of MCMC chains.
        target_accept_prob : float, optional
            Target acceptance probability for NUTS.
        dense_mass : bool, optional
            If True, use a dense mass matrix during NUTS adaptation.
        max_tree_depth : int, optional
            Maximum NUTS tree depth.
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

        psf_mags : object
            psf_mags value.
        psf_mag_errs : object
            psf_mag_errs value.
        psf_filter_curves : object
            psf_filter_curves value.
        use_psf_phot : object
            use_psf_phot value.
        custom_components : object
            custom_components value.
        custom_line_components : object
            custom_line_components value.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        custom_components = normalize_custom_components(custom_components)
        custom_line_components = normalize_custom_line_components(custom_line_components)
        if prior_config is None:
            prior_config = _materialize_prior_config(_build_default_prior_config(flux))
        else:
            prior_config = _materialize_prior_config(prior_config)
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
            tied_line_meta = build_tied_line_meta_from_linelist(
                line_table,
                wave,
                pool_narrow_centroids=bool(
                    prior_config.get("pool_narrow_centroids", True)
                ),
            )
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
                reddening_key = 'log_ebv' if 'log_ebv' in prior_config else 'log_reddening_a2500'
                if bool(prior_config.get("standardize_active_priors", False)):
                    init_vals[f'{reddening_key}_std'] = 0.0
                else:
                    init_vals[reddening_key] = prior_config.get(reddening_key, {}).get('loc', np.log(0.1))
            init_vals.update(
                _build_line_init_values(tied_line_meta, prior_config, use_lines=use_lines)
            )
        else:
            init_vals = dict(init_values)
            for key, value in _build_line_init_values(
                tied_line_meta, prior_config, use_lines=use_lines
            ).items():
                init_vals.setdefault(key, value)
        init_strategy = init_to_value(values=init_vals)
        reparam_config = _numpyro_geometry_reparam_config(
            prior_config,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_reddening=fit_reddening,
            fit_poly_order=fit_poly_order,
            decompose_host=decompose_host,
        )
        nuts_model = (
            reparam(qso_fsps_joint_model, config=reparam_config)
            if reparam_config
            else qso_fsps_joint_model
        )
        if bool(prior_config.get("line_block_dense_mass", False)) and use_lines:
            mass_matrix_structure = _line_complex_dense_mass_blocks(
                tied_line_meta,
                standardized_amplitudes=bool(
                    prior_config.get("standardize_active_priors", False)
                ),
            )
        else:
            mass_matrix_structure = bool(dense_mass)
        self.nuts_mass_matrix_structure = mass_matrix_structure
        kernel = NUTS(
            nuts_model,
            init_strategy=init_strategy,
            target_accept_prob=target_accept_prob,
            dense_mass=mass_matrix_structure,
            max_tree_depth=int(max_tree_depth),
        )
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True, jit_model_args=False)
        rng_key = jax.random.PRNGKey(int(random_seed))
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
            **self._static_component_cache_kwargs(),
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
            extra_fields=("num_steps", "accept_prob"),
        )
        # Always expose convergence diagnostics in both terminal sessions and
        # notebook cell output immediately after NumPyro sampling completes.
        mcmc.print_summary()
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
            **self._static_component_cache_kwargs(),
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

    def _plot_initialization(self, wave, flux, err, pred_out, samples, *, stage_name, attr_prefix, model_label):
        """Plot and store an Optax initialization model.

        Parameters
        ----------
        wave : object
            wave value.
        flux : object
            flux value.
        err : object
            err value.
        pred_out : object
            pred_out value.
        samples : object
            samples value.
        stage_name : object
            stage_name value.
        attr_prefix : object
            attr_prefix value.
        model_label : object
            model_label value.
        """
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

        setattr(self, f"{attr_prefix}_samples", samples)
        setattr(self, f"{attr_prefix}_pred_out", pred_out)
        setattr(self, f"{attr_prefix}_model", model)
        setattr(self, f"{attr_prefix}_continuum_model", continuum)
        setattr(self, f"{attr_prefix}_host_model", host)
        setattr(self, f"{attr_prefix}_pl_model", pl)
        setattr(self, f"{attr_prefix}_line_model", line)
        setattr(self, f"{attr_prefix}_redchi2", redchi2)

        from .plotting import plot_initialization

        plot_initialization(
            wave=wave,
            flux=flux,
            model=model,
            host=host,
            powerlaw=pl,
            line=line,
            redchi2=redchi2,
            stage_name=stage_name,
            model_label=model_label,
            show_plot=True,
        )

    def _plot_stage1_initialization(self, wave, flux, err, pred_out, samples):
        """Plot and store the stage-1 Optax continuum/host warm-start model.

        Parameters
        ----------
        wave : object
            wave value.
        flux : object
            flux value.
        err : object
            err value.
        pred_out : object
            pred_out value.
        samples : object
            samples value.
        """
        self._plot_initialization(
            wave,
            flux,
            err,
            pred_out,
            samples,
            stage_name="Stage 1 initialization",
            attr_prefix="init_stage1",
            model_label="stage 1 model",
        )

    def _plot_stage2_initialization(self, wave, flux, err, pred_out, samples):
        """Plot and store the full stage-2 Optax MAP model passed to NUTS.

        Parameters
        ----------
        wave : object
            wave value.
        flux : object
            flux value.
        err : object
            err value.
        pred_out : object
            pred_out value.
        samples : object
            samples value.
        """
        self._plot_initialization(
            wave,
            flux,
            err,
            pred_out,
            samples,
            stage_name="Stage 2 full MAP initialization",
            attr_prefix="init_stage2",
            model_label="stage 2 MAP model",
        )

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
                           plot_init=False, random_seed=0):
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

        psf_mags : object
            psf_mags value.
        psf_mag_errs : object
            psf_mag_errs value.
        psf_filter_curves : object
            psf_filter_curves value.
        use_psf_phot : object
            use_psf_phot value.
        custom_components : object
            custom_components value.
        custom_line_components : object
            custom_line_components value.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        custom_components = normalize_custom_components(custom_components)
        custom_line_components = normalize_custom_line_components(custom_line_components)
        if prior_config is None:
            prior_config = _materialize_prior_config(_build_default_prior_config(flux))
        else:
            prior_config = _materialize_prior_config(prior_config)
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
            tied_line_meta = build_tied_line_meta_from_linelist(
                line_table,
                wave,
                pool_narrow_centroids=bool(
                    prior_config.get("pool_narrow_centroids", True)
                ),
            )
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

        def _stage1_continuum_keep_mask(wave_in):
            """Mask strong optical emission-line windows for continuum warm start.

            Parameters
            ----------
            wave_in : object
                wave_in value.
            """
            line_windows = (
                (2700.0, 2900.0),   # Mg II
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
            """Return PSF filter curves restricted to a wavelength subset.

            Parameters
            ----------
            curves : object
                curves value.
            keep_mask : object
                keep_mask value.
            """
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
            """Run an SVI stage and return optimizer state/results.

            Parameters
            ----------
            guide : object
                guide value.
            steps : object
                steps value.
            use_lines_i : object
                use_lines_i value.
            fit_pl_i : object
                fit_pl_i value.
            fit_fe_i : object
                fit_fe_i value.
            fit_bc_i : object
                fit_bc_i value.
            fit_poly_i : object
                fit_poly_i value.
            fit_reddening_i : object
                fit_reddening_i value.
            fit_poly_order_i : object
                fit_poly_order_i value.
            decompose_host_i : object
                decompose_host_i value.
            wave_i : object
                wave_i value.
            flux_i : object
                flux_i value.
            err_i : object
                err_i value.
            fsps_grid_i : object
                fsps_grid_i value.
            psf_filter_curves_i : object
                psf_filter_curves_i value.
            """
            wave_run = wave if wave_i is None else wave_i
            flux_run = flux if flux_i is None else flux_i
            err_run = err if err_i is None else err_i
            fsps_grid_run = fsps_grid if fsps_grid_i is None else fsps_grid_i
            psf_filter_curves_run = psf_filter_curves if psf_filter_curves_i is None else psf_filter_curves_i
            optimizer = optax_to_numpyro(optax.adam(learning_rate))
            svi = SVI(qso_fsps_joint_model, guide, optimizer, loss=Trace_ELBO())
            key = jax.random.PRNGKey(int(random_seed))
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
                **self._static_component_cache_kwargs(),
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
            """Read a scalar field from a prior-config entry.

            Parameters
            ----------
            key : object
                key value.
            field : object
                field value.
            default : object
                default value.
            """
            cfg = prior_config.get(key, default)
            if isinstance(cfg, dict):
                value = cfg.get(field, cfg.get('value', cfg.get('loc', default)))
            elif isinstance(cfg, dist.Distribution):
                base_dist = getattr(cfg, 'base_dist', cfg)
                value = getattr(cfg, field, getattr(base_dist, field, getattr(base_dist, 'loc', default)))
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
                if fit_reddening and bool(prior_config.get("residualize_reddening_geometry", False)):
                    values['PL_apparent_log_norm_std'] = np.array(0.0)
                    values['PL_apparent_slope_std'] = np.array(0.0)
                elif bool(prior_config.get("standardize_active_priors", False)):
                    values['PL_norm_std'] = np.array(0.0)
                    values['PL_slope_std'] = np.array(0.0)
                else:
                    values['PL_norm'] = max(pl_init, 1e-8)
                if fit_reddening:
                    reddening_key = 'log_ebv' if 'log_ebv' in prior_config else 'log_reddening_a2500'
                    if bool(prior_config.get("standardize_active_priors", False)):
                        values[f'{reddening_key}_std'] = np.array(0.0)
                    else:
                        values[reddening_key] = _prior_field(reddening_key, 'loc', np.log(0.1))

            if decompose_host and host_sfh_model in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}:
                values['log_stellar_mass'] = _prior_field('log_stellar_mass', 'loc', 9.0)
                values['log_sfh_age_gyr'] = _prior_field('log_sfh_age_gyr', 'loc', np.log(3.0))
                values['log_sfh_tau_over_age'] = _prior_field('log_sfh_tau_over_age', 'loc', 0.0)
                gal_lgmet = _prior_field('gal_lgmet', 'loc', 0.0)
                ssp_lgmet = np.asarray(getattr(fsps_grid.host_basis_jax, 'ssp_lgmet', []), dtype=float)
                if ssp_lgmet.size and np.any(np.isfinite(ssp_lgmet)):
                    low = float(np.nanmin(ssp_lgmet))
                    high = float(np.nanmax(ssp_lgmet))
                    margin = min(1.0e-6, max(0.0, 0.25 * (high - low)))
                    gal_lgmet = float(np.clip(gal_lgmet, low + margin, high - margin))
                values['gal_lgmet'] = gal_lgmet
                values['log_gal_lgmet_scatter'] = _prior_field('log_gal_lgmet_scatter', 'loc', np.log(0.15))
                values['log_host_aperture_scale'] = _prior_field('log_host_aperture_scale', 'value', 0.0)
            return values

        # Stage 1: warm start on the continuum/host landscape. Keep reddening
        # active when requested so the stage-1 and stage-2 power-law latent
        # parameterizations match and the continuum MAP transfers correctly.
        n1 = max(100, int(num_steps // 3))
        stage1_keep = _stage1_continuum_keep_mask(wave)
        self.init_stage1_keep_mask = stage1_keep
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
            fit_reddening_i=fit_reddening,
            fit_poly_order_i=2,
            decompose_host_i=decompose_host,
            wave_i=wave[stage1_keep],
            flux_i=flux[stage1_keep],
            err_i=err[stage1_keep],
            fsps_grid_i=fsps_grid,
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
                **self._static_component_cache_kwargs(),
                use_lines=False,
                prior_config=prior_config,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=False,
                fit_bc=False,
                fit_poly=False,
                fit_reddening=fit_reddening,
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
        stage2_init_values = dict(map1)
        stage2_init_values.update(
            _build_line_init_values(tied_line_meta, prior_config, use_lines=use_lines)
        )
        guide2 = AutoDelta(
            qso_fsps_joint_model,
            init_loc_fn=init_to_value(values=stage2_init_values),
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
        rng_key = jax.random.PRNGKey(int(random_seed))

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
            **self._static_component_cache_kwargs(),
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
        if plot_init:
            self._plot_stage2_initialization(
                wave=wave,
                flux=flux,
                err=err,
                pred_out=pred_out,
                samples=samples,
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
                                dense_mass=True,
                                max_tree_depth=8,
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
                                plot_init=False, random_seed=0):
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
        dense_mass : bool, optional
            If True, use a dense mass matrix during NUTS adaptation.
        max_tree_depth : int, optional
            Maximum NUTS tree depth.
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

        psf_mags : object
            psf_mags value.
        psf_mag_errs : object
            psf_mag_errs value.
        psf_filter_curves : object
            psf_filter_curves value.
        use_psf_phot : object
            use_psf_phot value.
        custom_components : object
            custom_components value.
        custom_line_components : object
            custom_line_components value.
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
            random_seed=random_seed,
        )
        init_values = getattr(self, 'optax_map_point', None)
        self.run_fsps_numpyro_fit(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
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
            random_seed=random_seed,
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
        samples = dict(samples)
        for physical_site in (
            "PL_norm",
            "PL_slope",
            "frac_jitter",
            "frac_fe_jitter",
            "add_jitter",
            "line_amp_group",
        ):
            if physical_site not in samples and physical_site in pred_out:
                samples[physical_site] = np.asarray(pred_out[physical_site])
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
        def _finite_draw_summary(draws):
            """Return median/std without reducing empty or all-NaN draws."""
            draws = np.asarray(draws, dtype=float)
            finite = np.isfinite(draws)
            if draws.size == 0 or not np.any(finite):
                return np.nan, np.nan
            finite_draws = draws[finite]
            return float(np.median(finite_draws)), float(np.std(finite_draws))

        self.delta_m_psf, self.delta_m_psf_err = _finite_draw_summary(
            delta_m_draws
        )
        self.eta_psf, self.eta_psf_err = _finite_draw_summary(eta_psf_draws)
        self.scale_psf = 10.0 ** (-0.4 * self.delta_m_psf) if np.isfinite(self.delta_m_psf) else np.nan
        def _optional_draw_summary(key):
            """Return median/std for an optional predictive diagnostic.

            Parameters
            ----------
            key : object
                key value.
            """
            if key not in pred_out:
                return np.nan, np.nan
            return _finite_draw_summary(pred_out[key])

        self.host_redshift_prior_weight, self.host_redshift_prior_weight_err = _optional_draw_summary('host_redshift_prior_weight')
        self.host_redshift_prior_loc_eff, self.host_redshift_prior_loc_eff_err = _optional_draw_summary('host_redshift_prior_loc_eff')
        self.host_redshift_prior_scale_eff, self.host_redshift_prior_scale_eff_err = _optional_draw_summary('host_redshift_prior_scale_eff')
        self.host_redshift_prior_df_eff, self.host_redshift_prior_df_eff_err = _optional_draw_summary('host_redshift_prior_df_eff')

        def _band(x):
            """Compute 16th/84th percentile uncertainty band across samples.

            Parameters
            ----------
            x : object
                x value.
            """
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
        if 'ebv' in samples:
            ebv_med = float(np.nanmedian(np.asarray(samples['ebv'])))
            ebv_err = float(np.nanstd(np.asarray(samples['ebv'])))
        else:
            ebv_med = np.nan
            ebv_err = np.nan
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
            ('ebv', ebv_med, 'float'),
            ('ebv_err', ebv_err, 'float'),
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
        """Validate sky coordinates before Galactic dereddening.

        Parameters
        ----------
        ra : object
            ra value.
        dec : object
            dec value.
        """
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

    def _original_spec(self, wave, flux, err):
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
        """Return PSF-space host fraction at wavelength ``w0``.


        Parameters
        ----------
        w0 : object
            w0 value.
        """
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

        _state : object
            _state value.
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
            prior_config = _materialize_prior_config(_build_default_prior_config(np.asarray(self.flux, dtype=float)))
        else:
            prior_config = _materialize_prior_config(prior_config)
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
        """Compute a simple BI-like integral from a BAL model and reference model.

        Parameters
        ----------
        wave : object
            wave value.
        bal_sum : object
            bal_sum value.
        reference : object
            reference value.
        line_center : object
            line_center value.
        vmin : object
            vmin value.
        vmax : object
            vmax value.
        min_width : object
            min_width value.
        depth_threshold : object
            depth_threshold value.
        """
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

        Parameters
        ----------
        component_names : object
            component_names value.
        line_center : object
            line_center value.
        vmin : object
            vmin value.
        vmax : object
            vmax value.
        min_width : object
            min_width value.
        depth_threshold : object
            depth_threshold value.
        include_line_emission : object
            include_line_emission value.
        return_details : object
            return_details value.
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
        """Flatten posterior samples into labeled 1D series for diagnostics.

        Parameters
        ----------
        param_names : object
            param_names value.
        max_vector_elems : object
            max_vector_elems value.
        """
        from .plotting import posterior_series

        return posterior_series(self, param_names=param_names, max_vector_elems=max_vector_elems)

    @staticmethod
    def _build_result_arrays(entries):
        """Convert ``(name, value, type)`` entries into legacy result arrays.


        Parameters
        ----------
        entries : object
            entries value.
        """
        return (
            np.array([value for _, value, _ in entries], dtype=object),
            np.array([dtype for _, _, dtype in entries], dtype=object),
            np.array([name for name, _, _ in entries], dtype=object),
        )

    @staticmethod
    def _filter_half_width_angstrom(filt):
        """Return an approximate half-width for a photometric filter.


        Parameters
        ----------
        filt : object
            filt value.
        """
        from .plotting import filter_half_width_angstrom

        return filter_half_width_angstrom(filt)

    def _plot_filter_metadata(self, bands):
        """Return plotting metadata arrays for the requested photometric bands.


        Parameters
        ----------
        bands : object
            bands value.
        """
        from .plotting import plot_filter_metadata

        return plot_filter_metadata(self, bands)

    @staticmethod
    def _style_axis(ax, spine_lw=1.5):
        """Apply consistent axis styling.

        Parameters
        ----------
        ax : object
            ax value.
        spine_lw : object
            spine_lw value.
        """
        from .plotting import style_axis

        return style_axis(ax, spine_lw=spine_lw)

    def _synthetic_photometry_for_plot(self, model_attr='model_total'):
        """Return rest-frame synthetic photometry points for plotting, if available.

        Parameters
        ----------
        model_attr : object
            model_attr value.
        """
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
        """Plot posterior trace series for selected parameters.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` to include all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses
            ``self.output_path`` or the current directory.
        save_fig_name : str or None, optional
            Output filename override.
        show_plot : bool, optional
            If True, display the figure interactively with ``plt.show()``.
        """
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
        """Plot posterior projections with ``corner.corner``.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` to include all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        bins : int, optional
            Histogram bin count.
        max_points : int, optional
            Maximum posterior draws to plot.
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses
            ``self.output_path`` or the current directory.
        save_fig_name : str or None, optional
            Output filename override.
        show_plot : bool, optional
            If True, display the figure interactively with ``plt.show()``.
        """
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
        """Plot trace and/or corner diagnostics in a single convenience call.

        Parameters
        ----------
        do_trace : bool, optional
            If True, render the trace plot.
        do_corner : bool, optional
            If True, render the corner plot.
        param_names : list[str] | str | None, optional
            Parameter selector shared by both plots. Use ``'all'`` to include
            all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        corner_bins : int, optional
            Histogram bin count for the corner plot.
        corner_max_points : int, optional
            Maximum posterior draws to use in the corner plot.
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses
            ``self.output_path`` or the current directory.
        show_plot : bool, optional
            If True, display each enabled figure interactively with
            ``plt.show()``.
        """
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
        """Plot the fitted spectrum, model components, and residuals.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :meth:`plot_fig`.
        """
        from .plotting import plot_spectrum

        return plot_spectrum(self, **kwargs)

    def plot_fig(self, save_fig_path=None, broad_fwhm=1200, plot_legend=True, ylims=None, plot_residual=True, show_title=True,
                 plot_1sigma=True, sigma_alpha=0.12, show_plot=True, plot_psf_space=False, plot_intrinsic_powerlaw=False):
        """Plot data, model components, line decomposition, and residuals.

        Parameters
        ----------
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses
            ``self.output_path`` or the current directory.
        broad_fwhm : float, optional
            Reserved broad-line threshold kept for compatibility.
        plot_legend : bool, optional
            If True, draw a legend.
        ylims : tuple[float, float] or None, optional
            Optional y-axis limits for the spectrum panel.
        plot_residual : bool, optional
            If True, draw the residual panel below the spectrum.
        show_title : bool, optional
            Reserved title toggle kept for compatibility.
        plot_1sigma : bool, optional
            If True, draw 16th-84th percentile posterior bands where available.
        sigma_alpha : float, optional
            Alpha transparency for posterior uncertainty bands.
        show_plot : bool, optional
            If True, call ``plt.show()``.
        plot_psf_space : bool, optional
            If True, plot the PSF-space model/components instead of the
            fiber-scale model/components.
        plot_intrinsic_powerlaw : bool, optional
            If True, overlay the intrinsic AGN power law before tilt and edge
            corrections.
        """
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
