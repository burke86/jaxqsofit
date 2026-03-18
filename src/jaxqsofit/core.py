from __future__ import annotations

import os
import pickle
import glob

import extinction
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from speclite import filters as speclite_filters

import jax
import jax.numpy as jnp
import optax
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import optax_to_numpyro

from .defaults import build_default_prior_config
from .model import (
    C_KMS,
    _extract_line_table_from_prior_config,
    _get_sfd_query,
    _normalize_template_flux,
    _np_to_jnp,
    _spectrum_center_pivot,
    build_fsps_template_grid,
    build_tied_line_meta_from_linelist,
    qso_fsps_joint_model,
    reconstruct_posterior_components,
    unred,
)

_SDSS_PSF_BANDS = ("u", "g", "r", "i", "z")
_SDSS_FILTER_CACHE = None


def _get_sdss_filters():
    """Load SDSS filter curves once and return a band->response mapping."""
    global _SDSS_FILTER_CACHE
    if _SDSS_FILTER_CACHE is None:
        filters = speclite_filters.load_filters(*[f"sdss2010-{b}" for b in _SDSS_PSF_BANDS])
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

class QSOFit:
    def __init__(self, lam, flux, err=None, z=0.0, ra=-999, dec=-999, filename=None, output_path=None,
                 wdisp=None, psf_mags=None, psf_mag_errs=None, psf_bands=None):
        """Initialize a spectral fitting object with observed-frame inputs.

        Parameters
        ----------
        lam : array-like
            Observed-frame wavelength array in Angstrom.
        flux : array-like
            Observed-frame flux density array.
        err : array-like or float or None, optional
            Per-pixel 1-sigma uncertainty. If ``None``, a default of ``1e-6``
            is used.
        z : float, optional
            Source redshift.
        ra, dec : float, optional
            Sky coordinates in degrees for Galactic dereddening.
        filename : str or None, optional
            Basename used for saving result tables/figures. If ``None``,
            a filename is auto-generated from ``ra`` and ``dec``.
        output_path : str or None, optional
            Output directory for saved artifacts.
        wdisp : array-like or None, optional
            Optional wavelength dispersion vector (stored only).
        psf_mags, psf_mag_errs : array-like or None, optional
            Optional PSF photometry magnitudes and 1-sigma errors.
        psf_bands : sequence of str or None, optional
            Band labels associated with ``psf_mags``. If omitted, defaults to
            the first N SDSS bands.

        Notes
        -----
        Providing per-pixel `err` is strongly recommended for robust inference.
        If `err` is not provided, a small default uncertainty (`1e-6`) is used;
        in that case, fitted intrinsic-scatter terms absorb much of the noise model.

        Use keyword arguments to avoid ambiguity (for example `z=...`).
        """
        self.lam_in = np.asarray(lam, dtype=np.float64)
        self.flux_in = np.asarray(flux, dtype=np.float64)
        if err is None:
            self.err_in = np.full_like(self.flux_in, 1e-6, dtype=np.float64)
        else:
            err_arr = np.asarray(err, dtype=np.float64)
            if err_arr.ndim == 0:
                self.err_in = np.full_like(self.flux_in, float(err_arr), dtype=np.float64)
            else:
                self.err_in = err_arr
        self.z = z
        self.wdisp = wdisp
        self.ra = ra
        self.dec = dec
        self.install_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = output_path
        self.filename = self._resolve_filename(filename=filename, ra=ra, dec=dec)
        self.psf_mags = None if psf_mags is None else np.asarray(psf_mags, dtype=np.float64)
        self.psf_mag_errs = None if psf_mag_errs is None else np.asarray(psf_mag_errs, dtype=np.float64)
        self.psf_mags_raw = None if psf_mags is None else np.asarray(psf_mags, dtype=np.float64)
        self.psf_mag_errs_raw = None if psf_mag_errs is None else np.asarray(psf_mag_errs, dtype=np.float64)
        self.psf_mags_dered = None
        self.psf_mag_errs_dered = None
        self.psf_bands = None if psf_bands is None else list(psf_bands)
        if self.psf_bands is None and self.psf_mags is not None:
            self.psf_bands = ["u", "g", "r", "i", "z"][:len(self.psf_mags)]
        self.psf_filter_curves = None
        self.use_psf_phot = False
        self.ebv_mw = np.nan

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

    @staticmethod
    def _predictive_return_sites():
        """Return posterior predictive sites needed for summaries and plots."""
        return [
            'f_pl_model',
            'f_fe_mgii_model',
            'f_fe_balmer_model',
            'f_bc_model',
            'f_poly_model',
            'agn_model',
            'gal_model',
            'line_model_broad',
            'line_model_narrow',
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
            'line_model_psf',
            'psf_model',
        ]

    def _prepare_psf_photometry(
        self,
        wave_obs,
        psf_mags=None,
        psf_mag_errs=None,
        psf_bands=None,
        use_psf_phot=False,
        min_filter_coverage=0.97,
    ):
        """Validate PSF photometry and interpolate filters onto the observed wavelength grid."""
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
            filt_wave = _filter_wave_to_angstrom_array(filt.wavelength)
            filt_trans = np.asarray(filt.response, dtype=np.float64)
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
        """Return the default on-disk path for a saved posterior bundle."""
        out_name = f"{self.filename}_samples.pkl" if save_name is None else save_name
        out_dir = self.output_path if save_path is None else save_path
        if out_dir is None:
            out_dir = '.'
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, out_name)

    @staticmethod
    def _serialize_for_pickle(value):
        """Recursively convert model state into pickle-friendly Python objects."""
        if isinstance(value, dict):
            return {k: QSOFit._serialize_for_pickle(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(QSOFit._serialize_for_pickle(v) for v in value)
        if isinstance(value, (np.ndarray, np.generic)):
            return np.asarray(value)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return np.asarray(value)
        return value

    def save_posterior_bundle(self, save_name=None, save_path=None):
        """Persist the fitted object state needed to reload posterior results."""
        state = {}
        excluded = {"numpyro_mcmc", "svi", "svi_state"}
        for key, value in self.__dict__.items():
            if key in excluded:
                continue
            state[key] = self._serialize_for_pickle(value)

        out_file = self._posterior_bundle_path(save_name=save_name, save_path=save_path)
        with open(out_file, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved posterior bundle: {out_file}")
        return out_file

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
        """Load a saved posterior bundle, optionally recreate figures, and return a QSOFit object."""
        if save_name is not None:
            bundle_name = save_name
            bundle_dir = '.' if output_path is None else output_path
            bundle_path = os.path.join(bundle_dir, bundle_name)
            resolved_name = cls._resolve_filename(filename=filename)
        elif filename is not None:
            resolved_name = cls._resolve_filename(filename=filename)
            bundle_name = f"{resolved_name}_samples.pkl"
            bundle_dir = '.' if output_path is None else output_path
            bundle_path = os.path.join(bundle_dir, bundle_name)
        else:
            bundle_dir = '.' if output_path is None else output_path
            matches = sorted(glob.glob(os.path.join(bundle_dir, "*_samples.pkl")))
            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No posterior bundle found under: {bundle_dir}. "
                    "Pass filename=..., output_path=..., or save_name=... explicitly."
                )
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple posterior bundles found under: {bundle_dir}. "
                    "Pass filename=... or save_name=... explicitly."
                )
            bundle_path = matches[0]
            bundle_name = os.path.basename(bundle_path)
            resolved_name = bundle_name[: -len("_samples.pkl")] if bundle_name.endswith("_samples.pkl") else bundle_name

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Posterior bundle not found: {bundle_path}")

        with open(bundle_path, "rb") as fh:
            state = pickle.load(fh)

        obj = cls(
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
        obj.install_path = os.path.dirname(os.path.abspath(__file__))

        if plot_fig:
            plot_kwargs = {} if kwargs_plot is None else dict(kwargs_plot)
            if "show_plot" not in plot_kwargs:
                plot_kwargs["show_plot"] = False
            obj.plot_fig(**plot_kwargs)
        if plot_diagnostics:
            diag_kwargs = {} if diagnostics_kwargs is None else dict(diagnostics_kwargs)
            obj.plot_mcmc_diagnostics(**diag_kwargs)
        return obj

    def fit(self, name=None, deredden=True,
            wave_range=None, wave_mask=None, save_fits_name=None,
            fit_lines=True, save_result=True, plot_fig=True, save_fig=True,
            show_plot=False,
            decompose_host=True,
            fit_pl=True,
            fit_fe=True,
            fit_bc=False,
            fit_poly=True,
            fit_poly_order=2,
            fit_poly_edge_flex=True,
            mask_lya_forest=True,
            fit_method='optax+nuts',
            verbose=True,
            fsps_age_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
            fsps_logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
            prior_config=None,
            dsps_ssp_fn='tempdata.h5',
            nuts_warmup=50,
            nuts_samples=50,
            nuts_chains=1,
            nuts_target_accept=0.9,
            optax_steps=600,
            optax_lr=1e-2,
            psf_mags=None,
            psf_mag_errs=None,
            psf_bands=None,
            use_psf_phot=False,
            kwargs_plot=None):
        """Run end-to-end preprocessing, fitting, and optional plotting/saving.

        Parameters
        ----------
        name : str or None, optional
            Optional override basename used for figure/output naming.
        deredden : bool, optional
            If True, apply Galactic dereddening with dustmaps SFD.
        wave_range : tuple[float, float] or None, optional
            Rest-frame wavelength range to keep.
        wave_mask : array-like or None, optional
            Rest-frame wavelength intervals to mask.
        save_fits_name : str or None, optional
            Basename for saved result table.
        fit_lines : bool, optional
            Enable/disable emission-line model.
        save_result : bool, optional
            If True, write summary results to CSV.
        plot_fig : bool, optional
            If True, render decomposition figure.
        save_fig : bool, optional
            If True, save rendered figures.
        show_plot : bool, optional
            If True, call ``plt.show()`` for the decomposition figure.
            Default is ``False`` to avoid interactive pop-ups in notebook/pipeline runs.
        decompose_host : bool, optional
            Enable/disable host SPS decomposition.
        fit_pl : bool, optional
            Enable/disable AGN power-law component.
        fit_fe : bool, optional
            Enable/disable FeII components.
        fit_bc : bool, optional
            Enable/disable Balmer continuum.
        fit_poly : bool, optional
            Enable/disable multiplicative polynomial tilt.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt. Uses
            coefficients ``poly_c1`` through ``poly_cN``.
        fit_poly_edge_flex : bool, optional
            If True, enable localized blue/red exponential edge correction
            terms within the polynomial model.
        mask_lya_forest : bool, optional
            If True, mask pixels with rest-frame wavelength below Ly-alpha
            (1215.67 Angstrom) before fitting.
        fit_method : {'nuts', 'optax', 'optax+nuts'}, optional
            Fitting backend.
        verbose : bool, optional
            Verbose optimizer output where applicable.
        fsps_age_grid : sequence of float, optional
            SSP age grid in Gyr.
        fsps_logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary. If None, defaults are auto-built.
        dsps_ssp_fn : str, optional
            Path to DSPS SSP HDF5 template file.
        nuts_warmup, nuts_samples : int, optional
            NUTS warmup and posterior sample counts.
        nuts_chains : int, optional
            Number of MCMC chains.
        nuts_target_accept : float, optional
            Target accept probability for NUTS.
        optax_steps : int, optional
            Number of SVI/Optax warm-start steps.
        optax_lr : float, optional
            Learning rate for Optax Adam.
        psf_mags, psf_mag_errs : array-like or None, optional
            Optional PSF-aperture photometry magnitudes and 1-sigma errors.
        psf_bands : sequence of str or None, optional
            Band labels for ``psf_mags``. Defaults to SDSS ``ugriz`` order.
        use_psf_phot : bool, optional
            If True, add a PSF-photometry likelihood term and infer PSF/fiber
            scaling plus host leakage.
        kwargs_plot : dict or None, optional
            Extra keyword arguments passed to :meth:`plot_fig`.
        """

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
        self._fit_fit_poly = bool(fit_poly)
        self._fit_fit_poly_order = int(fit_poly_order)
        self._fit_fit_poly_edge_flex = bool(fit_poly_edge_flex)
        self._fit_mask_lya_forest = bool(mask_lya_forest)
        self._fit_method = str(fit_method)
        self._fit_fsps_age_grid = tuple(fsps_age_grid)
        self._fit_fsps_logzsol_grid = tuple(fsps_logzsol_grid)
        self._fit_prior_config = prior_config
        self._fit_dsps_ssp_fn = str(dsps_ssp_fn)
        self._fit_use_psf_phot = bool(use_psf_phot)

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

        if save_fits_name is None:
            save_fits_name = self.filename

        ind_gooderror = np.where((self.err_in > 0) & np.isfinite(self.err_in) & (self.flux_in != 0) & np.isfinite(self.flux_in), True, False)
        self.err = self.err_in[ind_gooderror]
        self.flux = self.flux_in[ind_gooderror]
        self.lam = self.lam_in[ind_gooderror]

        if prior_config_input is None:
            prior_config = build_default_prior_config(self.flux)
        out_params = prior_config.get('out_params', {})
        self.L_conti_wave = np.asarray(out_params.get('cont_loc', []), dtype=float)
        self._fit_prior_config = prior_config

        if wave_range is not None:
            self._wave_trim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._wave_msk(self.lam, self.flux, self.err, self.z)
        if mask_lya_forest:
            self._mask_lya_forest(self.lam, self.flux, self.err, self.z)
        if deredden:
            self._de_redden(self.lam, self.flux, self.err, self.ra, self.dec)

        self._rest_frame(self.lam, self.flux, self.err, self.z)
        self._calculate_sn(self.wave, self.flux)
        self._orignial_spec(self.wave, self.flux, self.err)
        psf_mags_use, psf_mag_errs_use, _psf_bands_use, psf_filter_curves_use, use_psf_phot_use = self._prepare_psf_photometry(
            wave_obs=self.lam,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_bands=psf_bands,
            use_psf_phot=use_psf_phot,
        )

        if fit_method == 'nuts':
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
                fit_poly_order=fit_poly_order,
                fit_poly_edge_flex=fit_poly_edge_flex,
                psf_mags=psf_mags_use,
                psf_mag_errs=psf_mag_errs_use,
                psf_filter_curves=psf_filter_curves_use,
                use_psf_phot=use_psf_phot_use,
            )
        elif fit_method == 'optax':
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
                fit_poly_order=fit_poly_order,
                fit_poly_edge_flex=fit_poly_edge_flex,
                psf_mags=psf_mags_use,
                psf_mag_errs=psf_mag_errs_use,
                psf_filter_curves=psf_filter_curves_use,
                use_psf_phot=use_psf_phot_use,
            )
        elif fit_method == 'optax+nuts':
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
                fit_poly_order=fit_poly_order,
                fit_poly_edge_flex=fit_poly_edge_flex,
                psf_mags=psf_mags_use,
                psf_mag_errs=psf_mag_errs_use,
                psf_filter_curves=psf_filter_curves_use,
                use_psf_phot=use_psf_phot_use,
            )
        else:
            raise ValueError(f"Unknown fit_method='{fit_method}'. Use 'nuts', 'optax', or 'optax+nuts'.")

        if save_result:
            self.save_result(self.conti_result, self.conti_result_type, self.conti_result_name,
                             self.line_result, self.line_result_type, self.line_result_name,
                             save_fits_name)
            self.save_posterior_bundle()
        if plot_fig:
            self.plot_fig(**kwargs_plot)

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
                             fit_poly_order=2,
                             fit_poly_edge_flex=True,
                             psf_mags=None,
                             psf_mag_errs=None,
                             psf_filter_curves=None,
                             use_psf_phot=False,
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
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_poly_edge_flex : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        init_values : dict or None, optional
            Optional initial values for ``init_to_value``.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        if prior_config is None:
            prior_config = build_default_prior_config(flux)
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None:
            raise ValueError(
                "fit_lines=True requires line priors/table in prior_config. "
                "Pass prior_config['line_priors'] (or prior_config['line']['table'])."
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
        fsps_grid = build_fsps_template_grid(
            wave_out=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
        )
        self.tied_line_meta = tied_line_meta

        init_vals = {'gal_v_kms': 0.0, 'gal_sigma_kms': 150.0} if init_values is None else init_values
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
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
            z_qso=self.z,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
        )
        samples = mcmc.get_samples()

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples=samples,
            return_sites=self._predictive_return_sites(),
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
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
            z_qso=self.z,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
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
                           fit_poly_order=2,
                           fit_poly_edge_flex=True,
                           psf_mags=None,
                           psf_mag_errs=None,
                           psf_filter_curves=None,
                           use_psf_phot=False):
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
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_poly_edge_flex : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        if prior_config is None:
            prior_config = build_default_prior_config(flux)
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None:
            raise ValueError(
                "fit_lines=True requires line priors/table in prior_config. "
                "Pass prior_config['line_priors'] (or prior_config['line']['table'])."
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
        fsps_grid = build_fsps_template_grid(
            wave_out=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
        )
        self.tied_line_meta = tied_line_meta

        def _run_svi(guide, steps, use_lines_i, fit_pl_i, fit_fe_i, fit_bc_i, fit_poly_i, fit_poly_order_i, fit_poly_edge_flex_i, decompose_host_i):
            """Run an SVI stage and return optimizer state/results."""
            optimizer = optax_to_numpyro(optax.adam(learning_rate))
            svi = SVI(qso_fsps_joint_model, guide, optimizer, loss=Trace_ELBO())
            key = jax.random.PRNGKey(0)
            result = svi.run(
                key,
                int(steps),
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
                use_lines=use_lines_i,
                prior_config=prior_config,
                decompose_host=decompose_host_i,
                fit_pl=fit_pl_i,
                fit_fe=fit_fe_i,
                fit_bc=fit_bc_i,
                fit_poly=fit_poly_i,
                fit_poly_order=fit_poly_order_i,
                fit_poly_edge_flex=fit_poly_edge_flex_i,
                z_qso=self.z,
                psf_mags=psf_mags,
                psf_mag_errs=psf_mag_errs,
                psf_filter_curves=psf_filter_curves,
                use_psf_phot=use_psf_phot,
                progress_bar=self.verbose,
            )
            return svi, result

        # Stage 1: warm start on simpler landscape (continuum/host only).
        n1 = max(100, int(num_steps // 3))
        guide1 = AutoDelta(
            qso_fsps_joint_model,
            init_loc_fn=init_to_value(values={'gal_v_kms': 0.0, 'gal_sigma_kms': 150.0}),
        )
        svi1, res1 = _run_svi(
            guide1,
            n1,
            use_lines_i=False,
            fit_pl_i=fit_pl,
            fit_fe_i=False,
            fit_bc_i=False,
            fit_poly_i=False,
            fit_poly_order_i=2,
            fit_poly_edge_flex_i=False,
            decompose_host_i=decompose_host,
        )
        map1 = guide1.median(res1.params)

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
            fit_poly_order_i=fit_poly_order,
            fit_poly_edge_flex_i=fit_poly_edge_flex,
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
            return_sites=self._predictive_return_sites(),
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
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
            z_qso=self.z,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
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
                                fit_poly_order=2,
                                fit_poly_edge_flex=True,
                                psf_mags=None,
                                psf_mag_errs=None,
                                psf_filter_curves=None,
                                use_psf_phot=False):
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
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_poly_edge_flex : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
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
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
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
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
            psf_mags=psf_mags,
            psf_mag_errs=psf_mag_errs,
            psf_filter_curves=psf_filter_curves,
            use_psf_phot=use_psf_phot,
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
        self.pred_out = pred_out
        self._pred_host_draws = np.asarray(pred_out['gal_model'])
        self._pred_bc_draws = np.asarray(pred_out['f_bc_model'])
        self._pred_cont_draws = np.asarray(pred_out['continuum_model'])
        self._pred_total_draws = np.asarray(pred_out['model'])
        self._pred_line_draws = np.asarray(pred_out['line_model'])
        self._pred_psf_draws = np.asarray(pred_out['psf_model']) if 'psf_model' in pred_out else None

        self.f_pl_model = np.median(np.asarray(pred_out['f_pl_model']), axis=0)
        self.f_fe_mgii_model = np.median(np.asarray(pred_out['f_fe_mgii_model']), axis=0)
        self.f_fe_balmer_model = np.median(np.asarray(pred_out['f_fe_balmer_model']), axis=0)
        self.f_bc_model = np.median(np.asarray(pred_out['f_bc_model']), axis=0)
        self.f_poly_model = np.median(np.asarray(pred_out['f_poly_model']), axis=0)
        self.qso = np.median(np.asarray(pred_out['agn_model']), axis=0)
        self.host = np.median(np.asarray(pred_out['gal_model']), axis=0)
        self.line_broad = np.median(np.asarray(pred_out['line_model_broad']), axis=0) if 'line_model_broad' in pred_out else np.full_like(self.qso, np.nan)
        self.line_narrow = np.median(np.asarray(pred_out['line_model_narrow']), axis=0) if 'line_model_narrow' in pred_out else np.full_like(self.qso, np.nan)
        self.f_line_model = np.median(np.asarray(pred_out['line_model']), axis=0)
        self.f_conti_model = np.median(np.asarray(pred_out['continuum_model']), axis=0)
        self.model_total = np.median(np.asarray(pred_out['model']), axis=0)
        self.qso_psf = np.median(np.asarray(pred_out['agn_model_psf']), axis=0) if 'agn_model_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.host_psf = np.median(np.asarray(pred_out['gal_model_psf']), axis=0) if 'gal_model_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.line_broad_psf = np.median(np.asarray(pred_out['line_model_broad_psf']), axis=0) if 'line_model_broad_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.line_narrow_psf = np.median(np.asarray(pred_out['line_model_narrow_psf']), axis=0) if 'line_model_narrow_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.line_psf = np.median(np.asarray(pred_out['line_model_psf']), axis=0) if 'line_model_psf' in pred_out else np.full_like(self.model_total, np.nan)
        self.psf_model = np.median(np.asarray(pred_out['psf_model']), axis=0) if 'psf_model' in pred_out else np.full_like(self.model_total, np.nan)
        self.fsps_weights_median = np.median(np.asarray(pred_out['fsps_weights']), axis=0)
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
        if self.verbose:
            print("max data        :", np.nanmax(self.flux))
            print("max total model :", np.nanmax(self.model_total))
            print("max PL          :", np.nanmax(self.f_pl_model))
            print("max host        :", np.nanmax(self.host))
            print("max FeII UV     :", np.nanmax(self.f_fe_mgii_model))
            print("max FeII opt    :", np.nanmax(self.f_fe_balmer_model))
            print("max Balmer cont :", np.nanmax(self.f_bc_model))
            print("max lines       :", np.nanmax(self.f_line_model))

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

        cont_waves = np.asarray(getattr(self, 'L_conti_wave', []), dtype=float)
        cont_waves = cont_waves[np.isfinite(cont_waves)]
        if cont_waves.size == 0:
            cont_waves = np.asarray([2500.0, 4200.0, 5100.0], dtype=float)
        self.L_conti_wave = cont_waves
        pivot_wave = float(np.asarray(_spectrum_center_pivot(self.wave), dtype=float))

        frac_host_vals = []
        frac_host_psf_vals = []
        frac_bc_vals = []
        frac_host_names = []
        frac_host_psf_names = []
        frac_bc_names = []
        for w0 in cont_waves:
            wave_label = self._format_wave_label(w0)
            frac_host = self._host_fraction_at_wave(w0)
            frac_host_psf = self._host_fraction_psf_at_wave(w0)
            frac_bc = self._bc_fraction_at_wave(w0)
            setattr(self, f'frac_host_{wave_label}', frac_host)
            setattr(self, f'frac_host_psf_{wave_label}', frac_host_psf)
            setattr(self, f'frac_bc_{wave_label}', frac_bc)
            frac_host_vals.append(frac_host)
            frac_host_psf_vals.append(frac_host_psf)
            frac_bc_vals.append(frac_bc)
            frac_host_names.append(f'frac_host_{wave_label}')
            frac_host_psf_names.append(f'frac_host_psf_{wave_label}')
            frac_bc_names.append(f'frac_bc_{wave_label}')

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
        if 'cont_norm' in samples:
            cont_samp = np.asarray(samples['cont_norm'])
            if decompose_host and 'log_frac_host' in samples:
                frac_host_samp = 1.0 / (1.0 + np.exp(-np.asarray(samples['log_frac_host'])))
                agn_samp = cont_samp * (1.0 - frac_host_samp)
            else:
                agn_samp = cont_samp
            pl_norm_samp = agn_samp
        elif 'PL_norm' in samples:
            pl_norm_samp = np.asarray(samples['PL_norm'])
        else:
            pl_norm_samp = np.full((n_samp,), np.nan)
        if 'PL_slope' in samples:
            pl_slope_med = float(np.nanmedian(np.asarray(samples['PL_slope'])))
            pl_slope_err = float(np.nanstd(np.asarray(samples['PL_slope'])))
        else:
            pl_slope_med = np.nan
            pl_slope_err = np.nan
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
            ('fsps_age_weighted_gyr', age_weighted, 'float'),
            ('fsps_logzsol_weighted', metal_weighted, 'float'),
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
        """
        if not hasattr(self, 'numpyro_samples') or not hasattr(self, 'fsps_grid'):
            raise RuntimeError("No posterior samples available. Run fit() first.")
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
            wave_out = np.arange(wmin, wmax + 0.5 * dw, dw, dtype=float)
        else:
            wave_out = np.asarray(wave_out, dtype=float)

        if wave_out.ndim != 1 or wave_out.size < 2 or not np.all(np.isfinite(wave_out)):
            raise ValueError("wave_out must be a finite 1D wavelength grid.")

        prior_config = getattr(self, '_fit_prior_config', None)
        if prior_config is None:
            prior_config = build_default_prior_config(np.asarray(self.flux, dtype=float))
        return reconstruct_posterior_components(
            wave_out=wave_out,
            samples=self.numpyro_samples,
            pred_out=getattr(self, 'pred_out', None),
            age_grid_gyr=getattr(self, '_fit_fsps_age_grid', self.fsps_grid.age_grid_gyr),
            logzsol_grid=getattr(self, '_fit_fsps_logzsol_grid', self.fsps_grid.logzsol_grid),
            dsps_ssp_fn=getattr(self, '_fit_dsps_ssp_fn', 'tempdata.h5'),
            prior_config=prior_config,
            fit_poly=bool(getattr(self, '_fit_fit_poly', False)),
            fit_poly_order=int(getattr(self, '_fit_fit_poly_order', 2)),
            fit_poly_edge_flex=bool(getattr(self, '_fit_fit_poly_edge_flex', False)),
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            n_draws=n_draws,
            return_components=return_components,
        )

    def component_fraction_at_wave(self, component='host', wave0=2500.0, reference='continuum', reconstruct=False, n_draws=None):
        """Return component/reference flux fraction at a requested wavelength.

        Parameters
        ----------
        component, reference : str, optional
            Component names. Supported reconstructed names are ``host``, ``PL``,
            ``Fe_uv``, ``Fe_op``, ``Balmer_cont``, ``edge_additive``, and ``continuum``.
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
        """Build a line-only profile from posterior-median Gaussian components.

        Parameters
        ----------
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        """
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
        if not hasattr(self, 'pred_out') or self.pred_out is None:
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
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` for all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        """
        if not hasattr(self, 'numpyro_samples') or self.numpyro_samples is None:
            return []

        samples = self.numpyro_samples
        if param_names == 'all':
            param_names = sorted(samples.keys())
        elif param_names is None:
            param_names = [
                'cont_norm', 'log_frac_host', 'PL_slope', 'Fe_uv_norm', 'Fe_op_norm',
                'Balmer_norm', 'Balmer_Te', 'Balmer_Tau',
                'gal_v_kms', 'gal_sigma_kms',
                'frac_jitter', 'add_jitter',
            ]

        out = []
        for name in param_names:
            if name not in samples:
                continue
            arr = np.asarray(samples[name])
            if arr.ndim == 1:
                out.append((name, arr))
            elif arr.ndim >= 2:
                arr2 = arr.reshape(arr.shape[0], -1)
                nflat = arr2.shape[1]
                if max_vector_elems is None or int(max_vector_elems) < 0:
                    ncomp = nflat
                else:
                    ncomp = min(nflat, int(max_vector_elems))
                for i in range(ncomp):
                    out.append((f'{name}[{i}]', arr2[:, i]))
        return out

    @staticmethod
    def _format_wave_label(w0):
        """Format a continuum wavelength for attribute/column naming."""
        try:
            w = float(w0)
        except Exception:
            return str(w0)
        if np.isfinite(w) and abs(w - round(w)) < 1e-6:
            return str(int(round(w)))
        return str(w).replace('.', 'p')

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
        filt_wave = _filter_wave_to_angstrom_array(filt.wavelength)
        filt_trans = np.asarray(filt.response, dtype=float)
        support = filt_wave[filt_trans > 0.01 * np.nanmax(filt_trans)]
        if support.size >= 2:
            return 0.5 * float(support.max() - support.min())
        return 0.0

    def _plot_filter_metadata(self, bands):
        """Return plotting metadata arrays for the requested photometric bands."""
        filters = _get_sdss_filters()
        filt_list = [filters.get(str(band)) for band in bands]
        valid = np.asarray([filt is not None for filt in filt_list], dtype=bool)
        filt_list = [filt for filt in filt_list if filt is not None]
        if len(filt_list) == 0:
            return valid, np.array([], dtype=float), np.array([], dtype=float)
        eff_wave_obs = np.asarray(
            [_filter_wave_to_angstrom_scalar(filt.effective_wavelength) for filt in filt_list],
            dtype=float,
        )
        half_width_obs = np.asarray(
            [self._filter_half_width_angstrom(filt) for filt in filt_list],
            dtype=float,
        )
        return valid, eff_wave_obs, half_width_obs

    @staticmethod
    def _style_axis(ax, spine_lw=1.5):
        """Apply consistent axis styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to style.
        spine_lw : float, optional
            Line width for all spines.
        """
        ax.grid(False)
        ax.tick_params(
            which='both',
            direction='in',
            top=True,
            right=True,
            length=6,
            width=1.2,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(spine_lw)

    def _synthetic_photometry_for_plot(self, model_attr='model_total'):
        """Return rest-frame synthetic photometry points for plotting, if available."""
        if not bool(getattr(self, 'use_psf_phot', False)):
            return None
        bands = list(getattr(self, 'psf_bands', []) or [])
        mag_errs = np.asarray(getattr(self, 'psf_mag_errs', []), dtype=float)
        if len(bands) == 0 or mag_errs.size != len(bands):
            return None
        if not hasattr(self, 'wave') or len(getattr(self, 'wave', [])) == 0:
            return None

        model_rf = np.asarray(getattr(self, model_attr, []), dtype=float)
        if model_rf.size != len(self.wave) or not np.any(np.isfinite(model_rf)):
            return None

        wave_rf = np.asarray(self.wave, dtype=float)
        z = float(getattr(self, 'z', 0.0))
        wave_obs = wave_rf * (1.0 + z)
        flam_obs = model_rf / max(1.0 + z, 1e-8)
        filters = _get_sdss_filters()
        c_ang_s = 2.99792458e18

        x_rf, xerr_rf, y_rf, yerr_rf = [], [], [], []
        for band, mag_err in zip(bands, mag_errs):
            filt = filters.get(str(band))
            if filt is None or not np.isfinite(mag_err) or mag_err <= 0:
                continue

            filt_wave = _filter_wave_to_angstrom_array(filt.wavelength)
            filt_trans = np.asarray(filt.response, dtype=float)
            trans = np.interp(wave_obs, filt_wave, filt_trans, left=0.0, right=0.0)
            trans = np.clip(trans, 0.0, None)
            flam_obs_cgs = 1e-17 * flam_obs
            num = float(np.trapezoid(flam_obs_cgs * trans * wave_obs, wave_obs))
            den = float(np.trapezoid(trans * c_ang_s / np.clip(wave_obs, 1e-8, None), wave_obs))
            if not np.isfinite(num) or not np.isfinite(den) or num <= 0 or den <= 0:
                continue

            mag_syn = -2.5 * np.log10(num / den) - 48.60
            if not np.isfinite(mag_syn):
                continue

            eff_wave_obs = _filter_wave_to_angstrom_scalar(filt.effective_wavelength)
            support = filt_wave[filt_trans > 0.01 * np.nanmax(filt_trans)]
            if support.size >= 2:
                half_width_obs = 0.5 * float(support.max() - support.min())
            else:
                half_width_obs = 0.0

            fnu_obs = 10.0 ** (-0.4 * (mag_syn + 48.60))
            flam_band_obs_cgs = fnu_obs * c_ang_s / max(eff_wave_obs ** 2, 1e-30)
            flam_band_obs = flam_band_obs_cgs / 1e-17
            flam_band_rf = flam_band_obs * (1.0 + z)
            flam_band_rf_err = flam_band_rf * (0.4 * np.log(10.0) * float(mag_err))

            x_rf.append(eff_wave_obs / (1.0 + z))
            xerr_rf.append(half_width_obs / (1.0 + z))
            y_rf.append(flam_band_rf)
            yerr_rf.append(flam_band_rf_err)

        if len(x_rf) == 0:
            return None
        return (
            np.asarray(x_rf, dtype=float),
            np.asarray(xerr_rf, dtype=float),
            np.asarray(y_rf, dtype=float),
            np.asarray(yerr_rf, dtype=float),
        )

    def _observed_photometry_for_plot(self):
        """Return rest-frame observed PSF photometry points for plotting, if available."""
        if not bool(getattr(self, 'use_psf_phot', False)):
            return None
        bands = list(getattr(self, 'psf_bands', []) or [])
        mags = np.asarray(getattr(self, 'psf_mags', []), dtype=float)
        mag_errs = np.asarray(getattr(self, 'psf_mag_errs', []), dtype=float)
        if len(bands) == 0 or mags.size != len(bands) or mag_errs.size != len(bands):
            return None

        z = float(getattr(self, 'z', 0.0))
        c_ang_s = 2.99792458e18
        filter_valid, eff_wave_obs, half_width_obs = self._plot_filter_metadata(bands)
        phot_valid = np.isfinite(mags) & np.isfinite(mag_errs) & (mag_errs > 0)
        valid = filter_valid & phot_valid
        if not np.any(valid):
            return None

        mags = mags[valid]
        mag_errs = mag_errs[valid]
        eff_wave_obs = eff_wave_obs[phot_valid[filter_valid]]
        half_width_obs = half_width_obs[phot_valid[filter_valid]]

        fnu_obs = 10.0 ** (-0.4 * (mags + 48.60))
        flam_band_obs_cgs = fnu_obs * c_ang_s / np.clip(eff_wave_obs ** 2, 1e-30, None)
        flam_band_obs = flam_band_obs_cgs / 1e-17
        flam_band_rf = flam_band_obs * (1.0 + z)
        flam_band_rf_err = flam_band_rf * (0.4 * np.log(10.0) * mag_errs)

        return (
            eff_wave_obs / (1.0 + z),
            half_width_obs / (1.0 + z),
            flam_band_rf,
            flam_band_rf_err,
        )

    def plot_trace(self, param_names=None, max_vector_elems=2, save_fig_path=None, save_fig_name=None):
        """Plot posterior trace series for selected parameters.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` to include all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        save_fig_name : str or None, optional
            Output filename override.
        """
        series = self._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
        if len(series) == 0:
            return None

        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(2.2 * n, 4)), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (label, vals) in zip(axes, series):
            ax.plot(np.arange(len(vals)), vals, color='tab:blue', lw=0.8)
            ax.set_ylabel(label, fontsize=9)
            self._style_axis(ax)
        axes[-1].set_xlabel('Sample', fontsize=10)
        fig.tight_layout()
        plt.show()
        if self.save_fig:
            out_name = f'{self.filename}_trace.pdf' if save_fig_name is None else save_fig_name
            save_dir = self.output_path if save_fig_path is None else save_fig_path
            if save_dir is None:
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, out_name)
            fig.savefig(out_file)
            print(f"Saved trace plot: {out_file}")
            plt.close(fig)
        self.trace_fig = fig
        return fig

    def plot_corner(self, param_names=None, max_vector_elems=2, bins=30, max_points=2000, save_fig_path=None, save_fig_name=None):
        """Plot a simple corner-style posterior projection matrix.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` to include all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        bins : int, optional
            Histogram bin count.
        max_points : int, optional
            Maximum posterior draws to plot (subsampled if needed).
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        save_fig_name : str or None, optional
            Output filename override.
        """
        series = self._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
        if len(series) == 0:
            return None

        labels = [s[0] for s in series]
        data = np.column_stack([s[1] for s in series])
        if data.shape[0] > int(max_points):
            idx = np.linspace(0, data.shape[0] - 1, int(max_points), dtype=int)
            data = data[idx]
        ndim = data.shape[1]

        fig, axes = plt.subplots(ndim, ndim, figsize=(2.2 * ndim, 2.2 * ndim))
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        for i in range(ndim):
            for j in range(ndim):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off')
                    continue
                if i == j:
                    ax.hist(data[:, j], bins=bins, color='tab:blue', alpha=0.75)
                else:
                    ax.hist2d(data[:, j], data[:, i], bins=bins, cmap='Blues', cmin=1)
                if i == ndim - 1:
                    ax.set_xlabel(labels[j], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=8)
                else:
                    ax.set_yticklabels([])
                self._style_axis(ax)
        fig.tight_layout(pad=0.35)
        plt.show()
        if self.save_fig:
            out_name = f'{self.filename}_corner.pdf' if save_fig_name is None else save_fig_name
            save_dir = self.output_path if save_fig_path is None else save_fig_path
            if save_dir is None:
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, out_name)
            fig.savefig(out_file)
            print(f"Saved corner plot: {out_file}")
            plt.close(fig)
        self.corner_fig = fig
        return fig

    def plot_mcmc_diagnostics(self, do_trace=True, do_corner=True,
                              param_names=None,
                              max_vector_elems=2,
                              corner_bins=30, corner_max_points=2000,
                              save_fig_path=None):
        """Plot trace and/or corner diagnostics in a single convenience call.

        Parameters
        ----------
        do_trace : bool, optional
            If True, render trace plot.
        do_corner : bool, optional
            If True, render corner plot.
        param_names : list[str] | str | None
            Parameter selector shared by both trace and corner plots.
            Use ``'all'`` to include all posterior parameters.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        corner_bins : int, optional
            Histogram bin count for corner plot.
        corner_max_points : int, optional
            Maximum posterior draws to use in corner plot.
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        """
        if do_trace:
            self.plot_trace(
                param_names=param_names,
                max_vector_elems=max_vector_elems,
                save_fig_path=save_fig_path,
            )
        if do_corner:
            self.plot_corner(
                param_names=param_names,
                max_vector_elems=max_vector_elems,
                bins=corner_bins,
                max_points=corner_max_points,
                save_fig_path=save_fig_path,
            )

    def plot_fig(self, save_fig_path=None, broad_fwhm=1200, plot_legend=True, ylims=None, plot_residual=True, show_title=True,
                 plot_1sigma=True, sigma_alpha=0.12, show_plot=True, plot_psf_space=False):
        """Plot data, model components, line decomposition, and residuals.

        Parameters
        ----------
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        broad_fwhm : float, optional
            Reserved broad-line threshold (kept for compatibility).
        plot_legend : bool, optional
            If True, draw a legend.
        ylims : tuple[float, float] or None, optional
            Optional y-axis limits for the spectrum panel.
        plot_residual : bool, optional
            If True, draw residual panel below the spectrum.
        show_title : bool, optional
            Reserved title toggle kept for compatibility.
        plot_1sigma : bool, optional
            If True, draw 16-84% posterior bands for available components.
        sigma_alpha : float, optional
            Alpha transparency of 1-sigma bands.
        show_plot : bool, optional
            If True, call ``plt.show()``. If False, figure is created/saved without display.
        plot_psf_space : bool, optional
            If True, plot the PSF-space model/components. In this mode the
            residual panel is suppressed because the observed spectrum remains
            on the fiber scale.
        """
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        psf_total_model = np.asarray(getattr(self, 'psf_model', []), dtype=float)
        use_psf_space = bool(plot_psf_space) and psf_total_model.size == len(getattr(self, 'wave', [])) and np.any(np.isfinite(psf_total_model))
        residual_enabled = bool(plot_residual) and not use_psf_space

        if residual_enabled:
            fig, (ax, ax_resid) = plt.subplots(
                2,
                1,
                figsize=(15, 8),
                sharex=True,
                gridspec_kw={'height_ratios': [4.0, 1.2], 'hspace': 0.05},
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax_resid = None

        flux_ref = float(np.nanpercentile(np.abs(self.flux[np.isfinite(self.flux)]), 95)) if np.any(np.isfinite(self.flux)) else 1.0
        comp_floor = max(1e-8, 0.005 * flux_ref)
        psf_scale = float(getattr(self, 'scale_psf', np.nan))
        psf_scale = psf_scale if np.isfinite(psf_scale) else 1.0

        total_model_plot = self.psf_model if use_psf_space else self.model_total
        host_plot = self.host_psf if use_psf_space else self.host
        pl_plot = psf_scale * self.f_pl_model if use_psf_space else self.f_pl_model
        fe_total_model = psf_scale * (self.f_fe_mgii_model + self.f_fe_balmer_model) if use_psf_space else (self.f_fe_mgii_model + self.f_fe_balmer_model)
        bc_plot = psf_scale * self.f_bc_model if use_psf_space else self.f_bc_model
        line_plot = self.line_psf if use_psf_space else self.f_line_model
        total_model_label = 'total model (PSF)' if use_psf_space else 'total model'
        host_label = 'host galaxy (PSF)' if use_psf_space else 'host galaxy'
        powerlaw_label = 'power law (PSF)' if use_psf_space else 'power law'
        fe_label = 'Fe II (PSF)' if use_psf_space else 'Fe II'
        bc_label = 'Balmer continuum (PSF)' if use_psf_space else 'Balmer continuum'
        line_label = 'total lines (PSF)' if use_psf_space else 'total lines'

        def _show_component(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return False
            return float(np.nanmax(np.abs(arr))) >= comp_floor

        if plot_1sigma and hasattr(self, 'pred_bands') and not use_psf_space:
            band_colors = {
                'total_model': 'b',
                'host': 'purple',
                'PL': 'orange',
                'FeII': 'teal',
                'Balmer_cont': 'y',
                'lines': 'lightskyblue',
            }
            for key, color in band_colors.items():
                if key not in self.pred_bands:
                    continue
                lo, hi = self.pred_bands[key]
                if len(lo) == len(self.wave) and _show_component(0.5 * (np.asarray(lo) + np.asarray(hi))):
                    ax.fill_between(
                        self.wave,
                        lo,
                        hi,
                        color=color,
                        alpha=sigma_alpha,
                        linewidth=0,
                        zorder=0,
                        rasterized=True,
                    )

        ax.plot(
            self.wave_prereduced,
            self.flux_prereduced,
            color='k' if not use_psf_space else 'gray',
            lw=1,
            label='data' if not use_psf_space else 'fiber data',
            zorder=2,
            alpha=1.0 if not use_psf_space else 0.6,
            rasterized=True,
        )
        ax.plot(self.wave, total_model_plot, color='b', lw=1.8, label=total_model_label, zorder=6, rasterized=True)
        if _show_component(host_plot):
            ax.plot(self.wave, host_plot, color='purple', lw=1.8, label=host_label, zorder=4, rasterized=True)
        else:
            ax.plot(self.wave, host_plot, color='purple', lw=1.8, zorder=4, rasterized=True)
        if _show_component(pl_plot):
            ax.plot(self.wave, pl_plot, color='orange', lw=1.5, label=powerlaw_label, zorder=5, rasterized=True)
        else:
            ax.plot(self.wave, pl_plot, color='orange', lw=1.5, zorder=5, rasterized=True)
        if _show_component(fe_total_model):
            ax.plot(self.wave, fe_total_model, color='teal', lw=1.2, label=fe_label, zorder=5, rasterized=True)
        else:
            ax.plot(self.wave, fe_total_model, color='teal', lw=1.2, zorder=5, rasterized=True)
        if _show_component(bc_plot):
            ax.plot(self.wave, bc_plot, color='y', lw=1.2, label=bc_label, zorder=5, rasterized=True)
        else:
            ax.plot(self.wave, bc_plot, color='y', lw=1.2, zorder=5, rasterized=True)
        if len(line_plot) == len(self.wave):
            if _show_component(line_plot):
                ax.plot(
                    self.wave,
                    line_plot,
                    color='lightskyblue',
                    lw=1.5,
                    label=line_label,
                    zorder=5,
                    rasterized=True,
                )
            else:
                ax.plot(
                    self.wave,
                    line_plot,
                    color='lightskyblue',
                    lw=1.5,
                    label=line_label,
                    zorder=5,
                    rasterized=True,
                )

        obs_phot_points = self._observed_photometry_for_plot()
        if obs_phot_points is not None:
            phot_x, phot_xerr, phot_y, phot_yerr = obs_phot_points
            ax.errorbar(
                phot_x,
                phot_y,
                yerr=phot_yerr,
                xerr=phot_xerr,
                fmt='s',
                ms=7,
                color='black',
                mfc='white',
                mec='black',
                mew=0.8,
                elinewidth=1.0,
                capsize=3,
                alpha=0.95,
                zorder=8,
                label='PSF photometry',
            )

        syn_phot_points = self._synthetic_photometry_for_plot(
            model_attr='psf_model' if use_psf_space else 'model_total'
        )
        if syn_phot_points is not None:
            phot_x, phot_xerr, phot_y, phot_yerr = syn_phot_points
            ax.errorbar(
                phot_x,
                phot_y,
                yerr=phot_yerr,
                xerr=phot_xerr,
                fmt='o',
                ms=7,
                color='crimson',
                mec='white',
                mew=0.8,
                elinewidth=1.0,
                capsize=3,
                alpha=0.95,
                zorder=8,
                label='synthetic photometry',
            )

        # Plot individual Gaussian line components: broad (*_br) in red, narrow in green.
        if (hasattr(self, 'line_component_amp_median')
                and hasattr(self, 'line_component_mu_median')
                and hasattr(self, 'line_component_sig_median')
                and hasattr(self, 'tied_line_meta')
                and len(self.line_component_amp_median) > 0):
            lnwave = np.log(self.wave)
            comp_labels = self.tied_line_meta.get('names', [''] * len(self.line_component_amp_median))
            drew_broad_label = False
            drew_narrow_label = False
            show_line_leg = _show_component(line_plot)
            for i in range(len(self.line_component_amp_median)):
                amp = float(self.line_component_amp_median[i])
                mu = float(self.line_component_mu_median[i])
                sig = float(self.line_component_sig_median[i])
                if not np.isfinite(amp) or not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
                    continue
                prof = amp * np.exp(-0.5 * ((lnwave - mu) / sig) ** 2)
                # Keep component plotting consistent with polynomial correction if enabled.
                if hasattr(self, 'f_poly_model') and len(self.f_poly_model) == len(prof):
                    prof = prof * self.f_poly_model
                cname = str(comp_labels[i]).lower()
                is_broad = cname.endswith('_br') or ('_br' in cname)
                if use_psf_space:
                    line_scale = psf_scale if is_broad else psf_scale * float(getattr(self, 'eta_psf', 1.0))
                    prof = line_scale * prof
                if is_broad:
                    lbl = 'broad components' if (show_line_leg and not drew_broad_label) else None
                    ax.plot(
                        self.wave,
                        prof,
                        color='red',
                        lw=0.7,
                        alpha=0.35,
                        zorder=3,
                        label=lbl,
                        rasterized=True,
                    )
                    drew_broad_label = True
                else:
                    lbl = 'narrow components' if (show_line_leg and not drew_narrow_label) else None
                    ax.plot(
                        self.wave,
                        prof,
                        color='green',
                        lw=0.7,
                        alpha=0.25,
                        zorder=3,
                        label=lbl,
                        rasterized=True,
                    )
                    drew_narrow_label = True

        ax.set_xlim(self.wave.min(), self.wave.max())
        if ylims is None:
            yplot = np.concatenate([
                self.flux[np.isfinite(self.flux)],
                total_model_plot[np.isfinite(total_model_plot)]
            ])
            if yplot.size > 0:
                y1, y2 = np.nanpercentile(yplot, [1, 99])
                if np.isfinite(y1) and np.isfinite(y2) and y2 > y1:
                    pad = 0.15 * (y2 - y1)
                    ax.set_ylim(0, y2 + pad)
        else:
            ax.set_ylim(0, ylims[1])

        # Mark common broad-line AGN transitions on the spectrum panel.
        broad_line_markers = [
            ("Ly$\\alpha$", 1215.67),
            ("CIV", 1549.06),
            ("CIII]", 1908.73),
            ("MgII", 2798.75),
            ("H$\\beta$", 4862.68),
            ("H$\\alpha$", 6564.61),
        ]
        xlo, xhi = ax.get_xlim()
        y_top = ax.get_ylim()[1]
        text_x_offset = 0.01 * (xhi - xlo)
        for label, lam0 in broad_line_markers:
            if xlo <= lam0 <= xhi:
                ax.axvline(lam0, color="gray", ls="--", lw=0.8, alpha=0.35, zorder=1)
                ax.text(
                    lam0 - text_x_offset,
                    y_top * 0.985,
                    label,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=12,
                    color="dimgray",
                    alpha=0.9,
                    zorder=7,
                )

        if residual_enabled and len(total_model_plot) == len(self.wave) and ax_resid is not None:
            resid = self.flux - total_model_plot
            ax_resid.plot(
                self.wave,
                resid,
                color='gray',
                ls='dotted',
                lw=1.0,
                zorder=2,
                rasterized=True,
            )
            ax_resid.axhline(0.0, color='k', ls='--', lw=0.8, zorder=1)
            r = resid[np.isfinite(resid)]
            if r.size > 0:
                rlim = np.nanpercentile(np.abs(r), 99)
                if np.isfinite(rlim) and rlim > 0:
                    ax_resid.set_ylim(-1.15 * rlim, 1.15 * rlim)
            ax_resid.set_ylabel('resid', fontsize=20)
            self._style_axis(ax_resid)

        if residual_enabled and ax_resid is not None:
            ax_resid.set_xlabel(r'Rest Wavelength ($\AA$)', fontsize=20)
        else:
            ax.set_xlabel(r'Rest Wavelength ($\AA$)', fontsize=20)
        ax.set_ylabel(r'$f_{\lambda}\ (10^{-17}\ \mathrm{erg}\ \mathrm{s}^{-1}\ \mathrm{cm}^{-2}\ \AA^{-1})$', fontsize=20)
        self._style_axis(ax)
        if plot_legend:
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=12, ncol=2)
        if show_plot:
            plt.show()
        if self.save_fig:
            save_dir = self.output_path if save_fig_path is None else save_fig_path
            if save_dir is None:
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, self.filename + '.pdf')
            fig.savefig(out_file, dpi=150)
            print(f"Saved spectrum plot: {out_file}")
            plt.close(fig)
        self.fig = fig
        return
