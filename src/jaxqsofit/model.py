from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import extinction

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from dsps import load_ssp_templates
from dustmaps.sfd import SFDQuery
from .custom_components import (
    CustomComponentSpec,
    CustomLineComponentSpec,
    normalize_custom_components,
    normalize_custom_line_components,
)

warnings.filterwarnings("ignore")

C_KMS = 299792.458
_SFD_QUERY_CACHE: Dict[str, Any] = {}
_LUMINOSITY_H0 = 70.0
_LUMINOSITY_OM0 = 0.3
MPC_TO_CM = 3.085677581491367e24


def unred(wave, flux, ebv, R_V=3.1):
    """Apply Fitzpatrick (1999) Galactic dereddening to a flux array."""
    # Preserve legacy function signature; use extinction package implementation.
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    a_lambda = extinction.fitzpatrick99(wave, a_v=R_V * ebv, r_v=R_V)
    return extinction.remove(a_lambda, flux)


def _np_to_jnp(x):
    """Convert an array-like object to float64 JAX array."""
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _normalize_template_flux(flux: np.ndarray, target_amp: float = 1.0) -> np.ndarray:
    """Rescale a template so its robust peak amplitude is O(target_amp)."""
    f = np.asarray(flux, dtype=float)
    robust = np.nanpercentile(np.abs(f), 99)
    if not np.isfinite(robust) or robust <= 0:
        robust = 1.0
    return f * (target_amp / robust)


def _spectrum_center_pivot(wave):
    """Use the midpoint of the fitted wavelength range as the power-law pivot."""
    wave = jnp.asarray(wave)
    return jnp.maximum(0.5 * (wave[0] + wave[-1]), 1e-8)


def _resolve_pl_pivot(wave, prior_config):
    """Return the configured power-law pivot or fall back to the spectrum center."""
    if prior_config is not None:
        pivot = prior_config.get("PL_pivot", None)
        if pivot is not None:
            return jnp.maximum(jnp.asarray(float(pivot)), 1e-8)
    return _spectrum_center_pivot(wave)


def _format_wave_label(w0):
    """Format a continuum wavelength for deterministic site naming."""
    try:
        wave = float(w0)
    except Exception:
        return str(w0)
    if np.isfinite(wave) and abs(wave - round(wave)) < 1e-6:
        return str(int(round(wave)))
    return str(wave).replace(".", "p")


def _continuum_output_waves_from_prior_config(prior_config, *, default_waves=(2500.0, 4200.0, 5100.0)):
    """Return unique continuum output wavelengths, always preserving 2500 A."""
    out_params = prior_config.get("out_params", {}) if isinstance(prior_config, dict) else {}
    waves = np.asarray(out_params.get("cont_loc", []), dtype=float)
    waves = waves[np.isfinite(waves)]
    if waves.size == 0:
        waves = np.asarray(default_waves, dtype=float)
    waves = np.concatenate([waves, np.asarray([2500.0], dtype=float)])

    out = []
    for wave in waves:
        wave = float(wave)
        if not np.isfinite(wave):
            continue
        if any(abs(wave - prev) < 1e-6 for prev in out):
            continue
        out.append(wave)
    return tuple(out)


@lru_cache(maxsize=256)
def _luminosity_distance_cm(z: float) -> float:
    """Return luminosity distance in cm for a fixed flat LCDM cosmology."""
    z = float(z)
    grid = np.linspace(0.0, max(z, 1.0e-8), 256, dtype=float)
    ez_inv = 1.0 / np.sqrt(np.maximum(_LUMINOSITY_OM0 * (1.0 + grid) ** 3 + (1.0 - _LUMINOSITY_OM0), 1.0e-18))
    dc_mpc = (C_KMS / _LUMINOSITY_H0) * np.trapezoid(ez_inv, x=grid)
    return float(dc_mpc * (1.0 + z) * MPC_TO_CM)


def _ez_inv_flat_lcdm_jax(z):
    """Inverse expansion rate for the fixed flat LCDM helper cosmology."""
    z = jnp.asarray(z, dtype=jnp.float64)
    ez2 = _LUMINOSITY_OM0 * (1.0 + z) ** 3 + (1.0 - _LUMINOSITY_OM0)
    return jax.lax.rsqrt(jnp.maximum(ez2, 1.0e-18))


def _luminosity_distance_cm_jax(z):
    """Return luminosity distance in cm using a pure-JAX flat LCDM integral."""
    z = jnp.asarray(z, dtype=jnp.float64)
    scalar_input = z.ndim == 0

    def _one_distance(zi):
        grid = jnp.linspace(0.0, jnp.maximum(zi, 1.0e-8), 256)
        dc_mpc = (C_KMS / _LUMINOSITY_H0) * jnp.trapezoid(_ez_inv_flat_lcdm_jax(grid), x=grid)
        return dc_mpc * (1.0 + zi) * MPC_TO_CM

    d_l_cm = _one_distance(z) if scalar_input else jax.vmap(_one_distance)(z)
    return jnp.reshape(d_l_cm, ()) if scalar_input else d_l_cm


def _rest_log_lambda_llambda_from_flam(wave_rest, flam_rest, z):
    """Return log10(lambda Llambda) using rest-frame f_lambda in 1e-17 cgs units."""
    wave_rest = jnp.maximum(jnp.asarray(wave_rest), 1e-8)
    flam_rest_cgs = 1e-17 * jnp.asarray(flam_rest)
    d_l_cm = _luminosity_distance_cm_jax(z)
    lambda_llambda = 4.0 * jnp.pi * d_l_cm**2 * wave_rest * flam_rest_cgs
    return jnp.log10(jnp.clip(lambda_llambda, 1e-300, None))


def _powerlaw_jax(wave, pl_norm, pl_slope, pivot):
    """Evaluate a power-law continuum at input wavelengths."""
    x = jnp.clip(wave / pivot, 1e-8, None)
    return pl_norm * x ** pl_slope


def _host_redshift_prior_params(prior_config, z_qso):
    """Return smooth redshift-dependent host prior weight, loc shift, scale multiplier, and df."""
    cfg = prior_config.get("host_redshift_prior", {}) if isinstance(prior_config, dict) else {}
    if not bool(cfg.get("enabled", True)):
        return jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0), None
    z_mid = jnp.asarray(float(cfg.get("z_mid", 1.0)))
    width = jnp.maximum(jnp.asarray(float(cfg.get("width", 0.2))), 1e-6)
    lowz_loc_offset = jnp.asarray(float(cfg.get("lowz_loc_offset", 0.0)))
    highz_loc_offset = jnp.asarray(float(cfg.get("highz_loc_offset", -8.0)))
    lowz_scale_mult = jnp.maximum(jnp.asarray(float(cfg.get("lowz_scale_mult", 1.0))), 1e-6)
    highz_scale_mult = jnp.maximum(jnp.asarray(float(cfg.get("highz_scale_mult", 0.05))), 1e-6)
    lowz_df = cfg.get("lowz_df", None)
    highz_df = cfg.get("highz_df", None)
    z_qso = jnp.asarray(z_qso)
    weight = jax.nn.sigmoid((z_qso - z_mid) / width)
    loc_offset = (1.0 - weight) * lowz_loc_offset + weight * highz_loc_offset
    scale_mult = (1.0 - weight) * lowz_scale_mult + weight * highz_scale_mult
    if lowz_df is None or highz_df is None:
        df_eff = None
    else:
        df_eff = (1.0 - weight) * jnp.asarray(float(lowz_df)) + weight * jnp.asarray(float(highz_df))
    return weight, loc_offset, scale_mult, df_eff


def negative_gaussian_bal_component(wave, params, metadata):
    """Additive negative BAL trough with optional super-Gaussian boxiness."""
    center = params["center"]
    sigma = jnp.maximum(params["sigma"], 1e-3)
    depth = jnp.maximum(params["depth"], 0.0)
    # ``shape_power=2`` reproduces the legacy Gaussian profile exactly.
    shape_power = jnp.maximum(params.get("shape_power", 2.0), 2.0)
    x = (wave - center) / sigma
    return -depth * jnp.exp(-0.5 * jnp.abs(x) ** shape_power)


def _smc_like_reddening_jax(wave, ebv, uv_ref=2500.0, alpha=1.2):
    """Return a smooth SMC-like attenuation curve."""
    ebv = jnp.maximum(jnp.asarray(ebv), 0.0)
    uv_ref = jnp.maximum(jnp.asarray(uv_ref), 1e-8)
    alpha = jnp.asarray(alpha)
    k_lambda = (jnp.clip(wave, 1e-8, None) / uv_ref) ** (-alpha)
    return 10.0 ** (-0.4 * ebv * k_lambda)


def _many_gauss_lnlam(lnlam, amps, mus, sigs):
    """Sum Gaussian components defined in log-wavelength space."""
    z = (lnlam[:, None] - mus[None, :]) / sigs[None, :]
    return jnp.sum(amps[None, :] * jnp.exp(-0.5 * z * z), axis=1)


def _line_meta_array(meta, key, *, jax_key=None, dtype=jnp.float64):
    """Return JAX-ready line metadata, preferring precomputed static arrays."""
    if jax_key is not None and jax_key in meta:
        return meta[jax_key]
    return jnp.asarray(meta[key], dtype=dtype)


def _broad_line_mask(names):
    """Return a float mask identifying broad-line components by name."""
    return np.asarray(
        [str(name).lower().endswith('_br') or ('_br' in str(name).lower()) for name in names],
        dtype=np.float64,
    )


def _synth_ab_mag_from_grid(wave_obs, flam_obs, filt_trans):
    """Compute an AB magnitude from flux density and filter transmission on one grid."""
    c_ang_s = 2.99792458e18
    trans = jnp.clip(filt_trans, 0.0, None)
    # Model spectra are stored in SDSS-style 1e-17 flux-density units.
    flam_obs_cgs = 1e-17 * flam_obs
    num = jnp.trapezoid(flam_obs_cgs * trans * wave_obs, wave_obs)
    den = jnp.trapezoid(trans * c_ang_s / jnp.clip(wave_obs, 1e-8, None), wave_obs)
    fnu = num / jnp.maximum(den, 1e-30)
    return -2.5 * jnp.log10(jnp.clip(fnu, 1e-30, None)) - 48.60


def _shift_and_broaden_single_spectrum_lnlam(lnwave, spectrum, v_kms, sigma_kms):
    """Apply LOS velocity shift and Gaussian broadening to one spectrum."""
    dln = jnp.mean(jnp.diff(lnwave))
    sigma_ln = jnp.maximum(sigma_kms / C_KMS, 1e-5)
    sigma_pix = sigma_ln / jnp.maximum(dln, 1e-8)
    kern = _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=128)

    wave = jnp.exp(lnwave)
    shift_ln = v_kms / C_KMS
    shifted_wave = jnp.exp(lnwave - shift_ln)
    shifted = jnp.interp(shifted_wave, wave, spectrum, left=0.0, right=0.0)
    return _convolve_same_length(shifted, kern)


def _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=512):
    """Build a normalized 1D Gaussian convolution kernel with fixed max size."""
    sigma_pix = jnp.maximum(sigma_pix, 1e-3)
    x = jnp.arange(-max_half, max_half + 1, dtype=jnp.float64)
    half_dyn = jnp.maximum(3.0, jnp.ceil(radius_mult * sigma_pix))
    mask = jnp.abs(x) <= half_dyn
    k = jnp.exp(-0.5 * (x / sigma_pix) ** 2)
    k = jnp.where(mask, k, 0.0)
    return k / jnp.maximum(jnp.sum(k), 1e-30)


def _convolve_same_length(signal, kernel):
    """Convolve and center-crop so the output matches the signal length."""
    signal = jnp.asarray(signal)
    kernel = jnp.asarray(kernel)
    full = jnp.convolve(signal, kernel, mode='same')
    n = signal.shape[0]
    m = full.shape[0]
    start = jnp.maximum((m - n) // 2, 0)
    return jax.lax.dynamic_slice(full, (start,), (n,))


def _fe_template_component(wave, wave_template, flux_template, norm, fwhm_kms, shift_frac, base_fwhm_kms=900.0):
    """Generate a broadened and shifted Fe template contribution."""
    # Enforce physically non-negative Fe pseudo-continuum and model broadening in velocity space.
    flux_template = jnp.maximum(flux_template, 0.0)
    template_on_wave = jnp.interp(wave, wave_template, flux_template, left=0.0, right=0.0)

    fwhm_eff = jnp.sqrt(jnp.maximum(fwhm_kms**2 - base_fwhm_kms**2, 910.0**2 - base_fwhm_kms**2))
    sigma_kms = fwhm_eff / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    v_kms = C_KMS * shift_frac
    lnwave = jnp.log(wave)
    model = _shift_and_broaden_single_spectrum_lnlam(lnwave, template_on_wave, v_kms, sigma_kms)
    return norm * model


def _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel):
    """Compute Balmer continuum template with edge-normalized blackbody shape."""
    lam_be = 3646.0
    h = 6.62607015e-27
    c = 2.99792458e10
    kb = 1.380649e-16

    wave = jnp.asarray(wave)
    lam_cm = wave * 1e-8

    expo = jnp.clip((h * c) / (lam_cm * kb * balmer_te), 1e-9, 700.0)
    bb = (2.0 * h * c**2 / lam_cm**5) / jnp.expm1(expo)
    bb = bb * 1e-8 * jnp.pi

    # Normalize shape at Balmer edge so balmer_norm is a flux-like amplitude.
    lam_be_cm = lam_be * 1e-8
    expo_edge = jnp.clip((h * c) / (lam_be_cm * kb * balmer_te), 1e-9, 700.0)
    bb_edge = (2.0 * h * c**2 / lam_be_cm**5) / jnp.expm1(expo_edge)
    bb_edge = bb_edge * 1e-8 * jnp.pi
    bb = bb / jnp.maximum(bb_edge, 1e-30)

    tau = balmer_tau * (wave / lam_be) ** 3
    bc = balmer_norm * (1.0 - jnp.exp(-tau)) * bb
    bc = jnp.where(wave <= lam_be, bc, 0.0)

    lnwave = jnp.log(wave)
    dln = jnp.mean(jnp.diff(lnwave))
    sigma_ln = jnp.maximum(balmer_vel / C_KMS, 1e-5)
    sigma_pix = sigma_ln / jnp.maximum(dln, 1e-8)
    kernel = _gaussian_kernel1d(sigma_pix)
    bc_conv = _convolve_same_length(bc, kernel)
    return bc_conv


def _cfg_norm_from_prior_config(prior_config, key):
    """Read Normal/LogNormal `(loc, scale)` parameters from prior config."""
    cfg = prior_config[key]
    if isinstance(cfg, dict) and ('loc' in cfg and 'scale' in cfg):
        return jnp.asarray(cfg['loc']), jnp.asarray(cfg['scale'])
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return jnp.asarray(cfg[0]), jnp.asarray(cfg[1])
    return jnp.asarray(cfg['loc']), jnp.asarray(cfg['scale'])


def _sample_from_prior_config(key, cfg):
    """Sample one parameter from a lightweight prior config dictionary."""
    dist_name = str(cfg.get("dist", "Normal")).lower()
    if dist_name == "normal":
        return numpyro.sample(key, dist.Normal(jnp.asarray(cfg["loc"]), jnp.asarray(cfg["scale"])))
    if dist_name == "lognormal":
        return numpyro.sample(key, dist.LogNormal(jnp.asarray(cfg["loc"]), jnp.asarray(cfg["scale"])))
    if dist_name == "halfnormal":
        return numpyro.sample(key, dist.HalfNormal(jnp.asarray(cfg["scale"])))
    if dist_name == "truncatednormal":
        return numpyro.sample(
            key,
            dist.TruncatedNormal(
                loc=jnp.asarray(cfg["loc"]),
                scale=jnp.asarray(cfg["scale"]),
                low=jnp.asarray(cfg["low"]),
                high=jnp.asarray(cfg["high"]),
            ),
        )
    raise ValueError(f"Unsupported custom-component prior distribution: {cfg.get('dist')}")


def _evaluate_custom_component_jax(wave, samples_or_values, comp, sample_value):
    """Evaluate one custom component from a sample/value mapping."""
    params = {
        param_name: sample_value(samples_or_values, comp.site_name(param_name), default=0.0)
        for param_name in comp.parameter_priors
    }
    return jnp.asarray(comp.evaluate(wave, params, comp.metadata), dtype=jnp.float64)


def _evaluate_custom_line_component_jax(wave, samples_or_values, comp, sample_value):
    """Evaluate one custom line component from a sample/value mapping."""
    params = {
        param_name: sample_value(samples_or_values, comp.site_name(param_name), default=0.0)
        for param_name in comp.parameter_priors
    }
    return jnp.asarray(comp.evaluate(wave, params, comp.metadata), dtype=jnp.float64)


@dataclass
class FSPSTemplateGrid:
    """Container for interpolated SSP templates and their metadata."""
    wave: np.ndarray
    templates: np.ndarray
    template_meta: List[Dict[str, float]]
    age_grid_gyr: np.ndarray
    logzsol_grid: np.ndarray


def _map_logzsol_to_dsps_lgmet(logzsol_grid: Sequence[float], ssp_lgmet: np.ndarray) -> np.ndarray:
    """Map fitting metallicity grid to DSPS metallicity convention."""
    logzsol_grid = np.asarray(logzsol_grid, dtype=float)
    ssp_lgmet = np.asarray(ssp_lgmet, dtype=float)

    # DSPS metallicity grids are often log10(Z), while fitting grids are usually log10(Z/Zsun).
    # Select the transform that best matches the available DSPS metallicity grid.
    cand_direct = logzsol_grid
    cand_shifted = logzsol_grid + np.log10(0.019)

    def mismatch(cand):
        """Return mean nearest-neighbor mismatch to DSPS metallicity grid."""
        return np.mean([np.min(np.abs(ssp_lgmet - val)) for val in cand])

    return cand_direct if mismatch(cand_direct) <= mismatch(cand_shifted) else cand_shifted


def _get_sfd_query():
    """Return cached dustmaps SFD query object."""
    cache_key = "default"
    if cache_key in _SFD_QUERY_CACHE:
        return _SFD_QUERY_CACHE[cache_key]

    q = SFDQuery()
    _SFD_QUERY_CACHE[cache_key] = q
    return q


def build_fsps_template_grid(
    wave_out: np.ndarray,
    age_grid_gyr: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0),
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2),
    imf_type: int = 1,
    zcontinuous: int = 1,
    sfh: int = 0,
    dsps_ssp_fn: str = 'tempdata.h5',
) -> FSPSTemplateGrid:
    """Build a host-galaxy SSP template matrix on the observed wavelength grid."""
    # Parameters kept for API compatibility.
    _ = (imf_type, zcontinuous, sfh)

    # DSPS quickstart pattern:
    # from dsps import load_ssp_templates
    # ssp_data = load_ssp_templates(fn='tempdata.h5')
    ssp_data = load_ssp_templates(fn=dsps_ssp_fn)
    ssp_lgmet = np.asarray(ssp_data.ssp_lgmet, dtype=float)
    ssp_lg_age_gyr = np.asarray(ssp_data.ssp_lg_age_gyr, dtype=float)
    ssp_wave = np.asarray(ssp_data.ssp_wave, dtype=float)
    ssp_flux = np.asarray(ssp_data.ssp_flux, dtype=float)

    wave_out = np.asarray(wave_out, dtype=float)
    age_grid_gyr = np.asarray(age_grid_gyr, dtype=float)
    logzsol_grid = np.asarray(logzsol_grid, dtype=float)
    target_lg_age = np.log10(np.clip(age_grid_gyr, 1e-5, None))
    target_lgmet = _map_logzsol_to_dsps_lgmet(logzsol_grid, ssp_lgmet)

    tmpl = []
    meta = []
    for i_z, logz in enumerate(logzsol_grid):
        imet = int(np.argmin(np.abs(ssp_lgmet - target_lgmet[i_z])))
        for i_a, age in enumerate(age_grid_gyr):
            iage = int(np.argmin(np.abs(ssp_lg_age_gyr - target_lg_age[i_a])))
            spec_native = np.asarray(ssp_flux[imet, iage, :], dtype=float)
            spec_interp = np.interp(wave_out, ssp_wave, spec_native, left=0.0, right=0.0)
            norm = np.nanmedian(np.abs(spec_interp))
            if not np.isfinite(norm) or norm <= 0:
                norm = 1.0
            spec_interp = spec_interp / norm
            tmpl.append(spec_interp)
            meta.append({
                'tage_gyr': float(age),
                'logzsol': float(logz),
                'norm': float(norm),
                'dsps_lgmet': float(ssp_lgmet[imet]),
                'dsps_lg_age_gyr': float(ssp_lg_age_gyr[iage]),
            })

    templates = np.column_stack(tmpl)
    return FSPSTemplateGrid(
        wave=wave_out,
        templates=templates,
        template_meta=meta,
        age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
        logzsol_grid=np.asarray(logzsol_grid, dtype=float),
    )


def reconstruct_posterior_components(
    wave_out: np.ndarray,
    samples: Dict[str, Any],
    pred_out: Dict[str, Any] | None,
    age_grid_gyr: Sequence[float],
    logzsol_grid: Sequence[float],
    dsps_ssp_fn: str,
    prior_config: Dict[str, Any],
    fit_poly: bool,
    fit_poly_order: int,
    fit_reddening: bool,
    fe_uv_wave: np.ndarray,
    fe_uv_flux: np.ndarray,
    fe_op_wave: np.ndarray,
    fe_op_flux: np.ndarray,
    custom_components: Sequence[CustomComponentSpec] | None = None,
    n_draws: int | None = None,
    return_components: bool = True,
) -> Dict[str, Any]:
    """Rebuild posterior continuum components on an arbitrary rest-frame grid."""
    wave_out = np.asarray(wave_out, dtype=float)
    if wave_out.ndim != 1 or wave_out.size < 2 or not np.all(np.isfinite(wave_out)):
        raise ValueError("wave_out must be a finite 1D wavelength grid.")

    fsps_grid = build_fsps_template_grid(
        wave_out=wave_out,
        age_grid_gyr=age_grid_gyr,
        logzsol_grid=logzsol_grid,
        dsps_ssp_fn=dsps_ssp_fn,
    )
    templates = np.asarray(fsps_grid.templates, dtype=float)
    lnwave = np.log(wave_out)
    custom_components = normalize_custom_components(custom_components)

    n_total = int(np.asarray(next(iter(samples.values()))).shape[0]) if len(samples) > 0 else 0
    if n_total == 0:
        raise RuntimeError("Posterior samples are empty.")
    n_use = n_total if n_draws is None else max(1, min(int(n_draws), n_total))
    sl = slice(0, n_use)

    cont_norm = np.asarray(samples.get('cont_norm', np.zeros(n_total)), dtype=float)[sl]
    log_frac_host = np.asarray(samples.get('log_frac_host', np.full(n_total, -np.inf)), dtype=float)[sl]
    frac_host = 1.0 / (1.0 + np.exp(-log_frac_host))
    pl_norm = np.asarray(samples.get('PL_norm', np.zeros(n_total)), dtype=float)[sl]
    pl_slope = np.asarray(samples.get('PL_slope', np.zeros(n_total)), dtype=float)[sl]
    gal_v = np.asarray(samples.get('gal_v_kms', np.zeros(n_total)), dtype=float)[sl]
    gal_sigma = np.asarray(samples.get('gal_sigma_kms', np.full(n_total, 150.0)), dtype=float)[sl]

    if pred_out is not None and 'fsps_weights' in pred_out:
        fsps_weights = np.asarray(pred_out['fsps_weights'], dtype=float)[sl]
    else:
        fsps_weights = np.zeros((n_use, templates.shape[1]), dtype=float)

    fe_uv_norm = np.asarray(samples.get('Fe_uv_norm', np.zeros(n_total)), dtype=float)[sl]
    log_fe_op_over_uv = np.asarray(samples.get('log_Fe_op_over_uv', np.zeros(n_total)), dtype=float)[sl]
    fe_op_norm = fe_uv_norm * np.exp(log_fe_op_over_uv)
    fe_uv_fwhm = np.asarray(samples.get('Fe_uv_FWHM', np.full(n_total, 3000.0)), dtype=float)[sl]
    fe_op_fwhm = np.asarray(samples.get('Fe_op_FWHM', np.full(n_total, 3000.0)), dtype=float)[sl]
    fe_uv_shift = np.asarray(samples.get('Fe_uv_shift', np.zeros(n_total)), dtype=float)[sl]
    fe_op_shift = np.asarray(samples.get('Fe_op_shift', np.zeros(n_total)), dtype=float)[sl]
    balmer_norm = np.asarray(samples.get('Balmer_norm', np.zeros(n_total)), dtype=float)[sl]
    balmer_tau = np.asarray(samples.get('Balmer_Tau', np.full(n_total, 0.5)), dtype=float)[sl]
    balmer_vel = np.asarray(samples.get('Balmer_vel', np.full(n_total, 3000.0)), dtype=float)[sl]

    component_draws = {
        'host': np.zeros((n_use, wave_out.size), dtype=float),
        'PL': np.zeros((n_use, wave_out.size), dtype=float),
        'Fe_uv': np.zeros((n_use, wave_out.size), dtype=float),
        'Fe_op': np.zeros((n_use, wave_out.size), dtype=float),
        'Balmer_cont': np.zeros((n_use, wave_out.size), dtype=float),
        'continuum': np.zeros((n_use, wave_out.size), dtype=float),
    }
    for comp in custom_components:
        component_draws[comp.output_name] = np.zeros((n_use, wave_out.size), dtype=float)
    pl_pivot = float(np.asarray(_resolve_pl_pivot(wave_out, prior_config), dtype=float))
    reddening_ebv = np.asarray(samples.get('reddening_ebv', np.zeros(n_total)), dtype=float)[sl]
    reddening_uv_ref = float(prior_config.get('reddening_uv_ref', 2500.0))
    reddening_alpha = float(prior_config.get('reddening_alpha', 1.2))

    for i in range(n_use):
        host_intrinsic = templates @ fsps_weights[i]
        host_model = np.asarray(
            _shift_and_broaden_single_spectrum_lnlam(lnwave, host_intrinsic, gal_v[i], gal_sigma[i]),
            dtype=float,
        )

        pl_model = np.asarray(
            _powerlaw_jax(
                wave_out,
                pl_norm=pl_norm[i],
                pl_slope=pl_slope[i],
                pivot=pl_pivot,
            ),
            dtype=float,
        )
        if fit_reddening:
            pl_model = pl_model * np.asarray(
                _smc_like_reddening_jax(
                    wave_out,
                    reddening_ebv[i],
                    uv_ref=reddening_uv_ref,
                    alpha=reddening_alpha,
                ),
                dtype=float,
            )
        fe_uv_model = np.asarray(
            _fe_template_component(wave_out, fe_uv_wave, fe_uv_flux, fe_uv_norm[i], fe_uv_fwhm[i], fe_uv_shift[i]),
            dtype=float,
        )
        fe_op_model = np.asarray(
            _fe_template_component(wave_out, fe_op_wave, fe_op_flux, fe_op_norm[i], fe_op_fwhm[i], fe_op_shift[i]),
            dtype=float,
        )
        bc_model = np.asarray(
            _balmer_continuum_jax(wave_out, balmer_norm[i], 15000.0, balmer_tau[i], balmer_vel[i]),
            dtype=float,
        )
        custom_total = np.zeros_like(wave_out)
        for comp in custom_components:
            def _sample_value(samples_dict, key, default=0.0):
                val = float(np.asarray(samples_dict.get(key, np.full(n_total, default)), dtype=float)[sl][i])
                return val

            custom_model = np.asarray(
                _evaluate_custom_component_jax(wave_out, samples, comp, _sample_value),
                dtype=float,
            )
            component_draws[comp.output_name][i] = custom_model
            custom_total = custom_total + custom_model

        poly_model = np.ones_like(wave_out)
        if fit_poly:
            w0 = 0.5 * (wave_out[0] + wave_out[-1])
            x = (wave_out - w0) / max(w0, 1.0)
            poly_base = np.ones_like(wave_out)
            for k in range(1, fit_poly_order + 1):
                key = f'poly_c{k}'
                if key in samples:
                    poly_base = poly_base + float(np.asarray(samples[key], dtype=float)[sl][i]) * (x ** k)
            poly_model = np.clip(poly_base, 0.2, 5.0)

        host_model = host_model * poly_model
        pl_model = pl_model * poly_model
        fe_uv_model = fe_uv_model * poly_model
        fe_op_model = fe_op_model * poly_model
        bc_model = bc_model * poly_model
        custom_total = custom_total * poly_model
        for comp in custom_components:
            component_draws[comp.output_name][i] = component_draws[comp.output_name][i] * poly_model
        continuum_model = pl_model + fe_uv_model + fe_op_model + bc_model + custom_total + host_model

        component_draws['host'][i] = host_model
        component_draws['PL'][i] = pl_model
        component_draws['Fe_uv'][i] = fe_uv_model
        component_draws['Fe_op'][i] = fe_op_model
        component_draws['Balmer_cont'][i] = bc_model
        component_draws['continuum'][i] = continuum_model

    output_draws = component_draws if return_components else {'continuum': component_draws['continuum']}
    return {
        'wave': wave_out,
        'draws': output_draws,
        'median': {key: np.median(val, axis=0) for key, val in output_draws.items()},
    }


def _extract_line_table_from_prior_config(prior_config: Dict[str, Any] | None):
    """Extract line-table style priors from `prior_config` in supported layouts."""
    if prior_config is None:
        return None
    if 'line_priors' in prior_config:
        return prior_config['line_priors']
    if 'line_table' in prior_config:
        return prior_config['line_table']
    line_cfg = prior_config.get('line', None)
    if isinstance(line_cfg, dict):
        if 'table' in line_cfg:
            return line_cfg['table']
        if 'priors' in line_cfg:
            return line_cfg['priors']
    return None


def _compress_group_ids(ids: np.ndarray, labels: Sequence[str] | None = None) -> Tuple[np.ndarray, Dict[Any, int]]:
    """Compress sparse positive tie ids into contiguous group indices."""
    out = np.full(len(ids), -1, dtype=int)
    mapping: Dict[Any, int] = {}
    next_gid = 0
    for i, gid in enumerate(ids):
        gid = int(gid)
        if gid <= 0:
            continue
        key: Any = gid if labels is None else (str(labels[i]), gid)
        if key not in mapping:
            mapping[key] = next_gid
            next_gid += 1
        out[i] = mapping[key]
    return out, mapping


def build_tied_line_meta_from_linelist(linelist, wave):
    """Build tied-line metadata arrays used by the NumPyro line model."""
    def _to_records(obj):
        """Normalize line table inputs to `list[dict]` records."""
        # pandas.DataFrame
        if hasattr(obj, 'to_dict'):
            return obj.to_dict('records')
        # Astropy table / FITS recarray / numpy structured array
        if hasattr(obj, 'dtype') and getattr(obj.dtype, 'names', None):
            return [{name: row[name] for name in obj.dtype.names} for row in obj]
        if hasattr(obj, 'colnames'):
            return [{name: row[name] for name in obj.colnames} for row in obj]
        # list[dict]-like
        return list(obj)

    records = _to_records(linelist)
    rows = []
    wmin = float(np.min(wave))
    wmax = float(np.max(wave))
    for row in records:
        lam = float(row['lambda'])
        if lam > wmin and lam < wmax:
            rows.append(row)

    ln_lambda0 = []
    amp_init = []
    amp_min = []
    amp_max = []
    sig_init = []
    sig_min = []
    sig_max = []
    dmu_min = []
    dmu_max = []
    names = []
    line_lambda = []
    vindex = []
    windex = []
    findex = []
    fvalue = []
    compnames = []

    for row in rows:
        for i in range(int(row.get('ngauss', 1))):
            ln0 = np.log(float(row['lambda']))
            voff = float(row['voff'])
            dln = voff / C_KMS
            ln_lambda0.append(ln0)
            line_lambda.append(float(row['lambda']))
            amp_init.append(float(row['inisca']))
            amp_min.append(float(row['minsca']))
            amp_max.append(float(row['maxsca']))
            sig_init.append(max(float(row['inisig']), 1e-5))
            sig_min.append(max(float(row['minsig']), 1e-5))
            sig_max.append(max(float(row['maxsig']), 1e-5))
            dmu_min.append(-dln)
            dmu_max.append(+dln)
            linename = str(row.get('linename', f"line_{row['lambda']:.1f}"))
            names.append(f"{linename}_{i+1}")
            vindex.append(int(row['vindex']))
            windex.append(int(row['windex']))
            findex.append(int(row['findex']))
            fvalue.append(float(row['fvalue']))
            compnames.append(str(row.get('compname', linename)))

    ln_lambda0 = np.asarray(ln_lambda0, dtype=float)
    amp_init = np.asarray(amp_init, dtype=float)
    amp_min = np.asarray(amp_min, dtype=float)
    amp_max = np.asarray(amp_max, dtype=float)
    sig_init = np.asarray(sig_init, dtype=float)
    sig_min = np.asarray(sig_min, dtype=float)
    sig_max = np.asarray(sig_max, dtype=float)
    dmu_min = np.asarray(dmu_min, dtype=float)
    dmu_max = np.asarray(dmu_max, dtype=float)
    vindex = np.asarray(vindex, dtype=int)
    windex = np.asarray(windex, dtype=int)
    findex = np.asarray(findex, dtype=int)
    fvalue = np.asarray(fvalue, dtype=float)

    # Tie indices are local to each line complex in qsopar; include compname in the key
    # to avoid accidental cross-complex tying when index integers are reused.
    vgroup, _ = _compress_group_ids(vindex, compnames)
    next_gid = np.max(vgroup) + 1 if len(vgroup) and np.any(vgroup >= 0) else 0
    for i in range(len(vgroup)):
        if vgroup[i] < 0:
            vgroup[i] = next_gid
            next_gid += 1
    n_vgroups = int(np.max(vgroup)) + 1 if len(vgroup) else 0

    wgroup, _ = _compress_group_ids(windex, compnames)
    next_gid = np.max(wgroup) + 1 if len(wgroup) and np.any(wgroup >= 0) else 0
    for i in range(len(wgroup)):
        if wgroup[i] < 0:
            wgroup[i] = next_gid
            next_gid += 1
    n_wgroups = int(np.max(wgroup)) + 1 if len(wgroup) else 0

    fgroup, _ = _compress_group_ids(findex, compnames)
    flux_ratio = np.ones(len(fgroup), dtype=float)
    next_gid = np.max(fgroup) + 1 if len(fgroup) and np.any(fgroup >= 0) else 0
    for local_gid in sorted(set([g for g in fgroup if g >= 0])):
        members = np.where(fgroup == local_gid)[0]
        ref = members[0]
        ref_f = fvalue[ref] if fvalue[ref] != 0 else 1.0
        for m in members:
            flux_ratio[m] = fvalue[m] / ref_f if ref_f != 0 else 1.0
    for i in range(len(fgroup)):
        if fgroup[i] < 0:
            fgroup[i] = next_gid
            flux_ratio[i] = 1.0
            next_gid += 1
    n_fgroups = int(np.max(fgroup)) + 1 if len(fgroup) else 0

    amp_init_group = np.zeros(n_fgroups, dtype=float)
    amp_min_group = np.zeros(n_fgroups, dtype=float)
    amp_max_group = np.zeros(n_fgroups, dtype=float)
    for gid in range(n_fgroups):
        ref = np.where(fgroup == gid)[0][0]
        amp_init_group[gid] = amp_init[ref]
        amp_min_group[gid] = amp_min[ref]
        amp_max_group[gid] = amp_max[ref]

    dmu_init_group = np.zeros(n_vgroups, dtype=float)
    dmu_min_group = np.zeros(n_vgroups, dtype=float)
    dmu_max_group = np.zeros(n_vgroups, dtype=float)
    for gid in range(n_vgroups):
        members = np.where(vgroup == gid)[0]
        dmu_init_group[gid] = 0.0
        dmu_min_group[gid] = np.max(dmu_min[members])
        dmu_max_group[gid] = np.min(dmu_max[members])

    sig_init_group = np.zeros(n_wgroups, dtype=float)
    sig_min_group = np.zeros(n_wgroups, dtype=float)
    sig_max_group = np.zeros(n_wgroups, dtype=float)
    for gid in range(n_wgroups):
        members = np.where(wgroup == gid)[0]
        sig_init_group[gid] = np.median(sig_init[members])
        sig_min_group[gid] = np.max(sig_min[members])
        sig_max_group[gid] = np.min(sig_max[members])
        if sig_max_group[gid] <= sig_min_group[gid]:
            sig_max_group[gid] = max(sig_min_group[gid] * 1.1, sig_min_group[gid] + 1e-4)

    broad_mask = _broad_line_mask(names)

    return {
        'n_lines': len(ln_lambda0),
        'n_vgroups': n_vgroups,
        'n_wgroups': n_wgroups,
        'n_fgroups': n_fgroups,
        'ln_lambda0': _np_to_jnp(ln_lambda0),
        'vgroup': np.asarray(vgroup, dtype=int),
        'vgroup_jax': jnp.asarray(vgroup, dtype=jnp.int32),
        'wgroup': np.asarray(wgroup, dtype=int),
        'wgroup_jax': jnp.asarray(wgroup, dtype=jnp.int32),
        'fgroup': np.asarray(fgroup, dtype=int),
        'fgroup_jax': jnp.asarray(fgroup, dtype=jnp.int32),
        'flux_ratio': np.asarray(flux_ratio, dtype=float),
        'flux_ratio_jax': _np_to_jnp(flux_ratio),
        'dmu_init_group': np.asarray(dmu_init_group, dtype=float),
        'dmu_init_group_jax': _np_to_jnp(dmu_init_group),
        'dmu_min_group': np.asarray(dmu_min_group, dtype=float),
        'dmu_min_group_jax': _np_to_jnp(dmu_min_group),
        'dmu_max_group': np.asarray(dmu_max_group, dtype=float),
        'dmu_max_group_jax': _np_to_jnp(dmu_max_group),
        'sig_init_group': np.asarray(sig_init_group, dtype=float),
        'sig_init_group_jax': _np_to_jnp(sig_init_group),
        'sig_min_group': np.asarray(sig_min_group, dtype=float),
        'sig_min_group_jax': _np_to_jnp(sig_min_group),
        'sig_max_group': np.asarray(sig_max_group, dtype=float),
        'sig_max_group_jax': _np_to_jnp(sig_max_group),
        'amp_init_group': np.asarray(amp_init_group, dtype=float),
        'amp_init_group_jax': _np_to_jnp(amp_init_group),
        'amp_min_group': np.asarray(amp_min_group, dtype=float),
        'amp_min_group_jax': _np_to_jnp(amp_min_group),
        'amp_max_group': np.asarray(amp_max_group, dtype=float),
        'amp_max_group_jax': _np_to_jnp(amp_max_group),
        'broad_mask_jax': _np_to_jnp(broad_mask),
        'names': names,
        'compnames': compnames,
        'line_lambda': np.asarray(line_lambda, dtype=float),
    }


def qso_fsps_joint_model(wave, flux, err, conti_priors, tied_line_meta, fsps_grid,
                         fe_uv_wave, fe_uv_flux, fe_op_wave, fe_op_flux, use_lines=True,
                         prior_config=None, decompose_host=True, fit_pl=True, fit_fe=True, fit_bc=True, fit_poly=False,
                         fit_poly_order=2,
                         fit_reddening=False, z_qso=0.0, psf_mags=None, psf_mag_errs=None,
                         psf_filter_curves=None, use_psf_phot=False,
                         return_line_components=True,
                         emit_deterministics=True,
                         custom_components: Sequence[CustomComponentSpec] | None = None,
                         custom_line_components: Sequence[CustomLineComponentSpec] | None = None):
    """Joint AGN+host spectral forward model for NumPyro inference."""
    wave = _np_to_jnp(wave)
    flux = _np_to_jnp(flux)
    err = _np_to_jnp(err)
    lnwave = jnp.log(wave)
    templates = _np_to_jnp(fsps_grid.templates)
    fe_uv_wave = _np_to_jnp(fe_uv_wave)
    fe_uv_flux = _np_to_jnp(fe_uv_flux)
    fe_op_wave = _np_to_jnp(fe_op_wave)
    fe_op_flux = _np_to_jnp(fe_op_flux)
    z_qso = jnp.asarray(z_qso, dtype=jnp.float64)
    prior_config = {} if prior_config is None else prior_config
    custom_components = normalize_custom_components(custom_components)
    custom_line_components = normalize_custom_line_components(custom_line_components)
    use_psf_phot = (
        bool(use_psf_phot)
        and psf_mags is not None
        and psf_mag_errs is not None
        and psf_filter_curves is not None
    )
    return_line_components = bool(return_line_components)
    emit_deterministics = bool(emit_deterministics)
    _cfg_norm = lambda key: _cfg_norm_from_prior_config(prior_config, key)

    def _cfg_halfnorm(key, ref_scale=None):
        """Read HalfNormal scale from prior config."""
        cfg = prior_config[key]
        if isinstance(cfg, dict):
            if 'scale' in cfg:
                return jnp.asarray(cfg['scale'])
            if 'scale_mult_err' in cfg:
                return jnp.asarray(cfg['scale_mult_err'] * ref_scale)
        if isinstance(cfg, (tuple, list)) and len(cfg) >= 1:
            return jnp.asarray(cfg[0])
        return jnp.asarray(cfg)

    # Continuum amplitude + host fraction parameterization
    cont_norm = numpyro.sample('cont_norm', dist.LogNormal(*_cfg_norm('log_cont_norm')))
    if isinstance(prior_config.get('log_frac_host', None), dict) and ('df' in prior_config['log_frac_host']):
        log_frac_host_df = float(prior_config['log_frac_host']['df'])
    else:
        log_frac_host_df = float(prior_config.get('log_frac_host_df', 3.0))
    if decompose_host:
        log_frac_host_loc, log_frac_host_scale = _cfg_norm('log_frac_host')
        host_redshift_prior_weight, host_redshift_prior_loc_offset, host_redshift_prior_scale_mult, host_redshift_prior_df_eff = _host_redshift_prior_params(prior_config, z_qso)
        log_frac_host_loc_eff = log_frac_host_loc + host_redshift_prior_loc_offset
        log_frac_host_scale_eff = jnp.maximum(log_frac_host_scale * host_redshift_prior_scale_mult, 1e-6)
        log_frac_host_df_eff = jnp.asarray(log_frac_host_df) if host_redshift_prior_df_eff is None else jnp.maximum(host_redshift_prior_df_eff, 1e-6)
        log_frac_host = numpyro.sample(
            'log_frac_host',
            dist.StudentT(df=log_frac_host_df_eff, loc=log_frac_host_loc_eff, scale=log_frac_host_scale_eff),
        )
        if emit_deterministics:
            numpyro.deterministic('host_redshift_prior_weight', host_redshift_prior_weight)
            numpyro.deterministic('host_redshift_prior_loc_eff', log_frac_host_loc_eff)
            numpyro.deterministic('host_redshift_prior_scale_eff', log_frac_host_scale_eff)
            numpyro.deterministic('host_redshift_prior_df_eff', log_frac_host_df_eff)
        frac_host = jax.nn.sigmoid(log_frac_host)
    else:
        if emit_deterministics:
            numpyro.deterministic('host_redshift_prior_weight', jnp.asarray(0.0))
            numpyro.deterministic('host_redshift_prior_loc_eff', jnp.asarray(0.0))
            numpyro.deterministic('host_redshift_prior_scale_eff', jnp.asarray(1.0))
            numpyro.deterministic('host_redshift_prior_df_eff', jnp.asarray(float(log_frac_host_df)))
        frac_host = jnp.asarray(0.0)
    host_amp = cont_norm * frac_host
    pl_pivot = _resolve_pl_pivot(wave, prior_config)
    if fit_pl:
        pl_norm = numpyro.sample('PL_norm', dist.HalfNormal(_cfg_halfnorm('PL_norm')))
        pl_slope_loc, pl_slope_scale = _cfg_norm('PL_slope')
        pl_slope = numpyro.sample('PL_slope', dist.Normal(pl_slope_loc, pl_slope_scale))
        reddening_ebv = numpyro.sample('reddening_ebv', dist.HalfNormal(_cfg_halfnorm('reddening_ebv'))) if fit_reddening else jnp.asarray(0.0)
    else:
        pl_norm = jnp.asarray(0.0)
        pl_slope = jnp.asarray(0.0)
        reddening_ebv = jnp.asarray(0.0)
        if decompose_host:
            frac_host = jnp.asarray(1.0)

    if fit_fe:
        fe_uv_norm = numpyro.sample('Fe_uv_norm', dist.LogNormal(*_cfg_norm('log_Fe_uv_norm')))
        log_fe_op_over_uv = numpyro.sample('log_Fe_op_over_uv', dist.Normal(*_cfg_norm('log_Fe_op_over_uv')))
        fe_op_norm = fe_uv_norm * jnp.exp(log_fe_op_over_uv)
        fe_uv_fwhm = numpyro.sample('Fe_uv_FWHM', dist.LogNormal(*_cfg_norm('log_Fe_uv_FWHM')))
        fe_op_fwhm = numpyro.sample('Fe_op_FWHM', dist.LogNormal(*_cfg_norm('log_Fe_op_FWHM')))
        fe_uv_shift = numpyro.sample('Fe_uv_shift', dist.Normal(*_cfg_norm('Fe_uv_shift')))
        fe_op_shift = numpyro.sample('Fe_op_shift', dist.Normal(*_cfg_norm('Fe_op_shift')))
    else:
        fe_uv_norm = jnp.asarray(0.0)
        fe_op_norm = jnp.asarray(0.0)
        fe_uv_fwhm = jnp.asarray(3000.0)
        fe_op_fwhm = jnp.asarray(3000.0)
        fe_uv_shift = jnp.asarray(0.0)
        fe_op_shift = jnp.asarray(0.0)

    if fit_bc:
        balmer_norm = numpyro.sample('Balmer_norm', dist.LogNormal(*_cfg_norm_from_prior_config(prior_config, 'log_Balmer_norm')))
        balmer_te = jnp.asarray(15000.0)
        balmer_tau = numpyro.sample('Balmer_Tau', dist.LogNormal(*_cfg_norm_from_prior_config(prior_config, 'log_Balmer_Tau')))
        balmer_vel = numpyro.sample('Balmer_vel', dist.LogNormal(*_cfg_norm_from_prior_config(prior_config, 'log_Balmer_vel')))
    else:
        balmer_norm = jnp.asarray(0.0)
        balmer_te = jnp.asarray(15000.0)
        balmer_tau = jnp.asarray(0.5)
        balmer_vel = jnp.asarray(3000.0)

    if fit_pl:
        pl_model_intrinsic = _powerlaw_jax(
            wave,
            pl_norm=pl_norm,
            pl_slope=pl_slope,
            pivot=pl_pivot,
        )
    else:
        pl_model_intrinsic = jnp.zeros_like(wave)
    reddening_atten = (
        _smc_like_reddening_jax(
            wave,
            reddening_ebv,
            uv_ref=float(prior_config.get('reddening_uv_ref', 2500.0)),
            alpha=float(prior_config.get('reddening_alpha', 1.2)),
        )
        if fit_reddening else jnp.ones_like(wave)
    )
    if fit_fe:
        fe_uv_model_intrinsic = _fe_template_component(wave, fe_uv_wave, fe_uv_flux, fe_uv_norm, fe_uv_fwhm, fe_uv_shift)
        fe_op_model_intrinsic = _fe_template_component(wave, fe_op_wave, fe_op_flux, fe_op_norm, fe_op_fwhm, fe_op_shift)
    else:
        fe_uv_model_intrinsic = jnp.zeros_like(wave)
        fe_op_model_intrinsic = jnp.zeros_like(wave)
    if fit_bc:
        bc_model_intrinsic = _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel)
    else:
        bc_model_intrinsic = jnp.zeros_like(wave)
    pl_model = pl_model_intrinsic * reddening_atten
    fe_uv_model = fe_uv_model_intrinsic
    fe_op_model = fe_op_model_intrinsic
    bc_model = bc_model_intrinsic
    custom_models = {}
    custom_total_model = jnp.zeros_like(wave)
    for comp in custom_components:
        def _sample_value(sample_dict, key, default=0.0):
            cfg = prior_config.get(key, None)
            if cfg is None:
                return default
            return _sample_from_prior_config(key, cfg)

        custom_model_intrinsic = _evaluate_custom_component_jax(wave, prior_config, comp, _sample_value)
        custom_models[comp.output_name] = custom_model_intrinsic
        custom_total_model = custom_total_model + custom_model_intrinsic
    poly_model = jnp.ones_like(wave)
    if fit_poly:
        poly_order = int(max(fit_poly_order, 0))
        w0 = 0.5 * (wave[0] + wave[-1])
        x = (wave - w0) / jnp.maximum(w0, 1.0)
        # Global low-order tilt
        poly_base = jnp.ones_like(wave)
        for k in range(1, poly_order + 1):
            ck = numpyro.sample(f'poly_c{k}', dist.Normal(*_cfg_norm(f'poly_c{k}')))
            poly_base = poly_base + ck * (x ** k)

        poly_model = jnp.clip(poly_base, 0.2, 5.0)
        pl_model = pl_model * poly_model
        fe_uv_model = fe_uv_model * poly_model
        fe_op_model = fe_op_model * poly_model
        bc_model = bc_model * poly_model
        custom_models = {name: model * poly_model for name, model in custom_models.items()}
        custom_total_model = custom_total_model * poly_model
    agn_model = pl_model + fe_uv_model + fe_op_model + bc_model + custom_total_model

    log_lambda_llambda_agn = {}
    for wave_lum in _continuum_output_waves_from_prior_config(prior_config):
        if fit_pl:
            pl_flux_lum = _powerlaw_jax(
                jnp.asarray(wave_lum),
                pl_norm=pl_norm,
                pl_slope=pl_slope,
                pivot=pl_pivot,
            )
            if fit_reddening:
                pl_flux_lum = pl_flux_lum * _smc_like_reddening_jax(
                    jnp.asarray(wave_lum),
                    reddening_ebv,
                    uv_ref=float(prior_config.get('reddening_uv_ref', 2500.0)),
                    alpha=float(prior_config.get('reddening_alpha', 1.2)),
                )
            log_lambda_llambda_agn[wave_lum] = _rest_log_lambda_llambda_from_flam(
                wave_lum,
                pl_flux_lum,
                z_qso,
            )
        else:
            log_lambda_llambda_agn[wave_lum] = jnp.asarray(jnp.nan)
    ntemp = fsps_grid.templates.shape[1]
    if decompose_host:
        tau_host = numpyro.sample('tau_host', dist.HalfNormal(_cfg_halfnorm('tau_host')))
        tau_host_eff = jnp.maximum(tau_host, 1e-6)
        raw_w_loc, _ = _cfg_norm('raw_w')
        raw_w = numpyro.sample('fsps_weights_raw', dist.Normal(jnp.full((ntemp,), raw_w_loc), tau_host_eff))
        fsps_weights_frac = jax.nn.softmax(raw_w)
        fsps_weights = host_amp * fsps_weights_frac
        gal_v_kms = numpyro.sample('gal_v_kms', dist.Normal(*_cfg_norm('gal_v_kms')))
        gal_sigma_kms = numpyro.sample('gal_sigma_kms', dist.HalfNormal(_cfg_halfnorm('gal_sigma_kms')))
        gal_intrinsic = jnp.dot(templates, fsps_weights)
        gal_model_intrinsic = _shift_and_broaden_single_spectrum_lnlam(lnwave, gal_intrinsic, gal_v_kms, gal_sigma_kms)
    else:
        fsps_weights_frac = jnp.zeros((ntemp,))
        fsps_weights = jnp.zeros((ntemp,))
        gal_model_intrinsic = jnp.zeros_like(wave)

    custom_line_models = {}
    custom_line_broad_intrinsic = jnp.zeros_like(wave)
    custom_line_narrow_intrinsic = jnp.zeros_like(wave)
    for comp in custom_line_components:
        def _sample_line_value(sample_dict, key, default=0.0):
            cfg = prior_config.get(key, None)
            if cfg is None:
                return default
            return _sample_from_prior_config(key, cfg)

        custom_line_model = _evaluate_custom_line_component_jax(wave, prior_config, comp, _sample_line_value)
        custom_line_models[comp.output_name] = custom_line_model
        if comp.line_kind == 'broad':
            custom_line_broad_intrinsic = custom_line_broad_intrinsic + custom_line_model
        else:
            custom_line_narrow_intrinsic = custom_line_narrow_intrinsic + custom_line_model

    line_components_are_split = return_line_components or use_psf_phot
    if use_lines and tied_line_meta['n_lines'] > 0:
        n_v = tied_line_meta['n_vgroups']
        n_w = tied_line_meta['n_wgroups']
        n_f = tied_line_meta['n_fgroups']
        dmu_scale_mult = float(prior_config['line_dmu_scale_mult'])
        sig_scale_mult = float(prior_config['line_sig_scale_mult'])
        amp_scale_mult = float(prior_config['line_amp_scale_mult'])

        dmu_group = numpyro.sample(
            'line_dmu_group',
            dist.TruncatedNormal(
                loc=_line_meta_array(tied_line_meta, 'dmu_init_group', jax_key='dmu_init_group_jax'),
                scale=jnp.maximum(
                    dmu_scale_mult * (
                        _line_meta_array(tied_line_meta, 'dmu_max_group', jax_key='dmu_max_group_jax')
                        - _line_meta_array(tied_line_meta, 'dmu_min_group', jax_key='dmu_min_group_jax')
                    ),
                    1e-6,
                ),
                low=_line_meta_array(tied_line_meta, 'dmu_min_group', jax_key='dmu_min_group_jax'),
                high=_line_meta_array(tied_line_meta, 'dmu_max_group', jax_key='dmu_max_group_jax'),
            )
        ) if n_v > 0 else jnp.zeros((0,))

        sig_group = numpyro.sample(
            'line_sig_group',
            dist.TruncatedNormal(
                loc=jnp.clip(_line_meta_array(tied_line_meta, 'sig_init_group', jax_key='sig_init_group_jax'), 1e-5),
                scale=jnp.maximum(
                    sig_scale_mult * (
                        _line_meta_array(tied_line_meta, 'sig_max_group', jax_key='sig_max_group_jax')
                        - _line_meta_array(tied_line_meta, 'sig_min_group', jax_key='sig_min_group_jax')
                    ),
                    1e-6,
                ),
                low=jnp.clip(_line_meta_array(tied_line_meta, 'sig_min_group', jax_key='sig_min_group_jax'), 1e-5),
                high=jnp.clip(_line_meta_array(tied_line_meta, 'sig_max_group', jax_key='sig_max_group_jax'), 1e-5),
            )
        ) if n_w > 0 else jnp.zeros((0,))

        amp_group = numpyro.sample(
            'line_amp_group',
            dist.TruncatedNormal(
                loc=jnp.clip(_line_meta_array(tied_line_meta, 'amp_init_group', jax_key='amp_init_group_jax'), 1e-10),
                scale=jnp.maximum(
                    amp_scale_mult * (
                        _line_meta_array(tied_line_meta, 'amp_max_group', jax_key='amp_max_group_jax')
                        - _line_meta_array(tied_line_meta, 'amp_min_group', jax_key='amp_min_group_jax')
                    ),
                    1e-10,
                ),
                low=jnp.clip(_line_meta_array(tied_line_meta, 'amp_min_group', jax_key='amp_min_group_jax'), 1e-10),
                high=jnp.clip(_line_meta_array(tied_line_meta, 'amp_max_group', jax_key='amp_max_group_jax'), 1e-10),
            )
        ) if n_f > 0 else jnp.zeros((0,))

        vgroup = _line_meta_array(tied_line_meta, 'vgroup', jax_key='vgroup_jax', dtype=jnp.int32)
        wgroup = _line_meta_array(tied_line_meta, 'wgroup', jax_key='wgroup_jax', dtype=jnp.int32)
        fgroup = _line_meta_array(tied_line_meta, 'fgroup', jax_key='fgroup_jax', dtype=jnp.int32)
        dmu = dmu_group[vgroup]
        sigs = sig_group[wgroup]
        amps = amp_group[fgroup] * _line_meta_array(tied_line_meta, 'flux_ratio', jax_key='flux_ratio_jax')
        mus = tied_line_meta['ln_lambda0'] + dmu

        if line_components_are_split:
            broad_mask = _line_meta_array(tied_line_meta, 'broad_mask', jax_key='broad_mask_jax')
            line_model_broad_intrinsic = _many_gauss_lnlam(lnwave, amps * broad_mask, mus, sigs)
            line_model_narrow_intrinsic = _many_gauss_lnlam(lnwave, amps * (1.0 - broad_mask), mus, sigs)
            line_model_broad_intrinsic = line_model_broad_intrinsic + custom_line_broad_intrinsic
            line_model_narrow_intrinsic = line_model_narrow_intrinsic + custom_line_narrow_intrinsic
            line_model_intrinsic = line_model_broad_intrinsic + line_model_narrow_intrinsic
        else:
            line_model_intrinsic = _many_gauss_lnlam(lnwave, amps, mus, sigs) + custom_line_broad_intrinsic + custom_line_narrow_intrinsic
            line_model_broad_intrinsic = jnp.zeros_like(wave)
            line_model_narrow_intrinsic = jnp.zeros_like(wave)
        if emit_deterministics:
            numpyro.deterministic('line_amp_per_component', amps)
            numpyro.deterministic('line_mu_per_component', mus)
            numpyro.deterministic('line_sig_per_component', sigs)
    else:
        line_model_broad_intrinsic = custom_line_broad_intrinsic
        line_model_narrow_intrinsic = custom_line_narrow_intrinsic
        line_model_intrinsic = custom_line_broad_intrinsic + custom_line_narrow_intrinsic

    gal_model = gal_model_intrinsic
    line_model_broad = line_model_broad_intrinsic
    line_model_narrow = line_model_narrow_intrinsic
    line_model = line_model_intrinsic
    if fit_poly:
        gal_model = gal_model * poly_model
        if line_components_are_split:
            line_model_broad = line_model_broad * poly_model
            line_model_narrow = line_model_narrow * poly_model
            line_model = line_model_broad + line_model_narrow
        else:
            line_model = line_model * poly_model
        custom_line_models = {name: model * poly_model for name, model in custom_line_models.items()}

    frac_jitter = numpyro.sample('frac_jitter', dist.HalfNormal(_cfg_halfnorm('frac_jitter')))
    add_jitter = numpyro.sample('add_jitter', dist.HalfNormal(_cfg_halfnorm('add_jitter', ref_scale=jnp.mean(err))))

    continuum_model = agn_model + gal_model
    model = continuum_model + line_model
    sigma_tot = jnp.sqrt(err**2 + (frac_jitter * jnp.abs(model))**2 + add_jitter**2)
    fiber_model = model

    delta_m_psf = jnp.asarray(0.0)
    eta_psf = jnp.asarray(1.0)
    scale_psf = jnp.asarray(1.0)
    agn_model_psf = agn_model
    gal_model_psf = gal_model
    line_model_broad_psf = line_model_broad
    line_model_narrow_psf = line_model_narrow
    line_model_psf = line_model_broad_psf + line_model_narrow_psf
    psf_model = agn_model_psf + gal_model_psf + line_model_psf
    if use_psf_phot:
        delta_m_psf = numpyro.sample('delta_m_psf_raw', dist.Normal(0.0, 0.5))
        if decompose_host:
            eta_psf = numpyro.sample('eta_psf_raw', dist.Beta(2.0, 2.0))
        scale_psf = 10.0 ** (-0.4 * delta_m_psf)
        agn_model_psf = scale_psf * agn_model
        gal_model_psf = scale_psf * eta_psf * gal_model
        line_model_broad_psf = scale_psf * line_model_broad
        line_model_narrow_psf = scale_psf * eta_psf * line_model_narrow
        line_model_psf = line_model_broad_psf + line_model_narrow_psf
        psf_model = agn_model_psf + gal_model_psf + line_model_psf

        wave_obs = wave * (1.0 + z_qso)
        flam_psf_obs = psf_model / jnp.maximum(1.0 + z_qso, 1e-8)
        psf_mags = _np_to_jnp(psf_mags)
        psf_mag_errs = _np_to_jnp(psf_mag_errs)
        psf_filter_trans = _np_to_jnp(psf_filter_curves['trans'])
        sigma_phot_extra = numpyro.sample('sigma_phot_extra', dist.HalfNormal(0.05))
        for i in range(psf_filter_trans.shape[0]):
            m_syn = _synth_ab_mag_from_grid(wave_obs, flam_psf_obs, psf_filter_trans[i])
            sig = jnp.sqrt(psf_mag_errs[i] ** 2 + sigma_phot_extra ** 2)
            numpyro.sample(f'psf_mag_obs_{i}', dist.Normal(m_syn, sig), obs=psf_mags[i])

    if emit_deterministics:
        numpyro.deterministic('f_pl_model', pl_model)
        numpyro.deterministic('f_fe_mgii_model', fe_uv_model)
        numpyro.deterministic('f_fe_balmer_model', fe_op_model)
        numpyro.deterministic('f_bc_model', bc_model)
        numpyro.deterministic('f_poly_model', poly_model)
        for comp in custom_components:
            numpyro.deterministic(comp.deterministic_site_name, custom_models[comp.output_name])
        for comp in custom_line_components:
            numpyro.deterministic(comp.deterministic_site_name, custom_line_models[comp.output_name])
        numpyro.deterministic('agn_model', agn_model)
        numpyro.deterministic('gal_model_intrinsic', gal_model_intrinsic)
        numpyro.deterministic('gal_model', gal_model)
        numpyro.deterministic('line_model_broad_intrinsic', line_model_broad_intrinsic)
        numpyro.deterministic('line_model_narrow_intrinsic', line_model_narrow_intrinsic)
        numpyro.deterministic('line_model_intrinsic', line_model_intrinsic)
        numpyro.deterministic('line_model_broad', line_model_broad)
        numpyro.deterministic('line_model_narrow', line_model_narrow)
        numpyro.deterministic('line_model', line_model)
        numpyro.deterministic('continuum_model', continuum_model)
        numpyro.deterministic('model', model)
        numpyro.deterministic('delta_m_psf', delta_m_psf)
        numpyro.deterministic('eta_psf', eta_psf)
        numpyro.deterministic('scale_psf', scale_psf)
        numpyro.deterministic('agn_model_psf', agn_model_psf)
        numpyro.deterministic('gal_model_psf', gal_model_psf)
        numpyro.deterministic('line_model_broad_psf', line_model_broad_psf)
        numpyro.deterministic('line_model_narrow_psf', line_model_narrow_psf)
        numpyro.deterministic('line_model_psf', line_model_psf)
        numpyro.deterministic('psf_model', psf_model)
        numpyro.deterministic('frac_host', frac_host)
        for wave_lum, log_lambda_llambda_lum in log_lambda_llambda_agn.items():
            wave_label = _format_wave_label(wave_lum)
            numpyro.deterministic(
                f'log_lambda_Llambda_{wave_label}_agn',
                log_lambda_llambda_lum,
            )
        numpyro.deterministic('fsps_weights', fsps_weights)
        numpyro.deterministic('fsps_weights_frac', fsps_weights_frac)

    student_t_df = float(prior_config.get('student_t_df', 3.0))
    numpyro.sample('obs', dist.StudentT(df=student_t_df, loc=fiber_model, scale=sigma_tot), obs=flux)
