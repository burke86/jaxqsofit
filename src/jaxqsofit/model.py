from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import extinction
import numpyro

numpyro.enable_x64()

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from dsps import load_ssp_templates
from dustmaps.sfd import SFDQuery
from jaxsedfit.host import (
    HostBasisJax,
    build_host_basis_jax,
    build_host_state as build_jaxsedfit_host_state,
)
from .custom_components import (
    CustomComponentSpec,
    CustomLineComponentSpec,
    custom_component_param_site,
    normalize_custom_components,
    normalize_custom_line_components,
)

C_KMS = 299792.458
_SFD_QUERY_CACHE: Dict[str, Any] = {}
_LUMINOSITY_H0 = 70.0
_LUMINOSITY_OM0 = 0.3
MPC_TO_CM = 3.085677581491367e24
W_PER_A_TO_CGS_PER_A = 1.0e7
CGS_TO_JAXQSOFIT_FLUX = 1.0e17
AMPLITUDE_FLOOR = 1e-32


def _materialize_prior_mapping(prior_config):
    """Return a flat prior mapping for low-level model helpers.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    """
    if prior_config is None:
        return {}
    if isinstance(prior_config, Mapping):
        return dict(prior_config)
    if hasattr(prior_config, "to_mapping"):
        return dict(prior_config.to_mapping())
    return dict(prior_config)


def unred(wave, flux, ebv, R_V=3.1):
    """Apply Fitzpatrick (1999) Galactic dereddening to a flux array.

    Parameters
    ----------
    wave : object
        wave value.
    flux : object
        flux value.
    ebv : object
        ebv value.
    R_V : object
        R_V value.
    """
    # Preserve legacy function signature; use extinction package implementation.
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    a_lambda = extinction.fitzpatrick99(wave, a_v=R_V * ebv, r_v=R_V)
    return extinction.remove(a_lambda, flux)


def _np_to_jnp(x):
    """Convert an array-like object to float64 JAX array.

    Parameters
    ----------
    x : object
        x value.
    """
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def spectral_likelihood_weight_from_resolving_power(wave, resolving_power):
    """Return the resolution-element spectral likelihood weight.

    Parameters
    ----------
    wave : object
        wave value.
    resolving_power : object
        resolving_power value.
    """
    if resolving_power is None:
        return jnp.asarray(1.0, dtype=jnp.float64)
    try:
        resolving_power_value = float(resolving_power)
    except Exception:
        return jnp.asarray(1.0, dtype=jnp.float64)
    if not np.isfinite(resolving_power_value) or resolving_power_value <= 0.0:
        return jnp.asarray(1.0, dtype=jnp.float64)

    wave = jnp.asarray(wave, dtype=jnp.float64)
    if wave.size < 2:
        return jnp.asarray(1.0, dtype=jnp.float64)
    delta = jnp.abs(wave[1:] - wave[:-1])
    prev_delta = jnp.zeros_like(wave).at[1:].set(delta)
    next_delta = jnp.zeros_like(wave).at[:-1].set(delta)
    pixel_width = 0.5 * (prev_delta + next_delta)
    valid = jnp.isfinite(wave) & (wave > 0.0) & (pixel_width > 0.0)
    resolution_width = wave / resolving_power_value
    n_eff = jnp.sum(jnp.where(valid, pixel_width / jnp.maximum(resolution_width, 1.0e-30), 0.0))
    n_pix = jnp.sum(valid.astype(jnp.float64))
    return jnp.where(n_pix > 0.0, jnp.minimum(n_eff / n_pix, 1.0), jnp.asarray(1.0, dtype=jnp.float64))


def _normalize_template_flux(flux: np.ndarray, target_amp: float = 1.0) -> np.ndarray:
    """Rescale a template so its robust peak amplitude is O(target_amp).

    Parameters
    ----------
    flux : object
        flux value.
    target_amp : object
        target_amp value.
    """
    f = np.asarray(flux, dtype=float)
    robust = np.nanpercentile(np.abs(f), 99)
    if not np.isfinite(robust) or robust <= 0:
        robust = 1.0
    return f * (target_amp / robust)


def _spectrum_center_pivot(wave):
    """Use the midpoint of the fitted wavelength range as the power-law pivot.


    Parameters
    ----------
    wave : object
        wave value.
    """
    wave = jnp.asarray(wave)
    return jnp.maximum(0.5 * (wave[0] + wave[-1]), 1e-8)


def _resolve_pl_pivot(wave, prior_config):
    """Return the configured power-law pivot or fall back to the spectrum center.

    Parameters
    ----------
    wave : object
        wave value.
    prior_config : object
        prior_config value.
    """
    prior_config = _materialize_prior_mapping(prior_config)
    pivot = prior_config.get("PL_pivot", None)
    if pivot is not None:
        return jnp.maximum(jnp.asarray(float(pivot)), 1e-8)
    return _spectrum_center_pivot(wave)


def _resolve_poly_pivot(wave, prior_config, *, require_configured=False):
    """Return the polynomial pivot wavelength used by the fitted model.

    Parameters
    ----------
    wave : object
        wave value.
    prior_config : object
        prior_config value.
    require_configured : object
        require_configured value.
    """
    prior_config = _materialize_prior_mapping(prior_config)
    pivot = prior_config.get("poly_pivot", None)
    if pivot is not None:
        return jnp.maximum(jnp.asarray(float(pivot)), 1e-8)
    if require_configured:
        raise ValueError(
            "Posterior reconstruction with fitted polynomial coefficients requires "
            "prior_config['poly_pivot'] from the fitted wavelength grid."
        )
    return _spectrum_center_pivot(wave)


def _format_wave_label(w0):
    """Format a continuum wavelength for deterministic site naming.

    Parameters
    ----------
    w0 : object
        w0 value.
    """
    try:
        wave = float(w0)
    except Exception:
        return str(w0)
    if np.isfinite(wave) and abs(wave - round(wave)) < 1e-6:
        return str(int(round(wave)))
    return str(wave).replace(".", "p")


def _continuum_output_waves_from_prior_config(prior_config, *, default_waves=(2500.0, 4200.0, 5100.0)):
    """Return unique continuum output wavelengths, always preserving 2500 A.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    default_waves : object
        default_waves value.
    """
    prior_config = _materialize_prior_mapping(prior_config)
    out_params = prior_config.get("out_params", {})
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
    """Return luminosity distance in cm for a fixed flat LCDM cosmology.

    Parameters
    ----------
    z : object
        z value.
    """
    z = float(z)
    grid = np.linspace(0.0, max(z, 1.0e-8), 256, dtype=float)
    ez_inv = 1.0 / np.sqrt(np.maximum(_LUMINOSITY_OM0 * (1.0 + grid) ** 3 + (1.0 - _LUMINOSITY_OM0), 1.0e-18))
    dc_mpc = (C_KMS / _LUMINOSITY_H0) * np.trapezoid(ez_inv, x=grid)
    return float(dc_mpc * (1.0 + z) * MPC_TO_CM)


def _ez_inv_flat_lcdm_jax(z):
    """Inverse expansion rate for the fixed flat LCDM helper cosmology.

    Parameters
    ----------
    z : object
        z value.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    ez2 = _LUMINOSITY_OM0 * (1.0 + z) ** 3 + (1.0 - _LUMINOSITY_OM0)
    return jax.lax.rsqrt(jnp.maximum(ez2, 1.0e-18))


def _luminosity_distance_cm_jax(z):
    """Return luminosity distance in cm using a pure-JAX flat LCDM integral.

    Parameters
    ----------
    z : object
        z value.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    scalar_input = z.ndim == 0

    def _one_distance(zi):
        """Integrate the fixed flat-LCDM luminosity distance for one redshift.

        Parameters
        ----------
        zi : object
            zi value.
        """
        grid = jnp.linspace(0.0, jnp.maximum(zi, 1.0e-8), 256)
        dc_mpc = (C_KMS / _LUMINOSITY_H0) * jnp.trapezoid(_ez_inv_flat_lcdm_jax(grid), x=grid)
        return dc_mpc * (1.0 + zi) * MPC_TO_CM

    d_l_cm = _one_distance(z) if scalar_input else jax.vmap(_one_distance)(z)
    return jnp.reshape(d_l_cm, ()) if scalar_input else d_l_cm


def _cosmic_age_gyr(z: float) -> float:
    """Return cosmic age in Gyr for the fixed flat LCDM helper cosmology.

    Parameters
    ----------
    z : object
        z value.
    """
    z = max(float(z), 0.0)
    grid = np.geomspace(1.0 + z, 1.0e4, 2048, dtype=float)
    ez = np.sqrt(np.maximum(_LUMINOSITY_OM0 * grid**3 + (1.0 - _LUMINOSITY_OM0), 1.0e-18))
    integral = np.trapezoid(1.0 / (grid * ez), x=grid)
    h0_s = (_LUMINOSITY_H0 * 1.0e5) / MPC_TO_CM
    return float(integral / h0_s / (365.25 * 24.0 * 3600.0 * 1.0e9))


def _host_luminosity_w_a_to_rest_flux_units(host_rest_lum_w_a, z_qso):
    """Convert rest L_lambda in W/A to JAXQSOFit rest-frame flux units.

    Parameters
    ----------
    host_rest_lum_w_a : object
        host_rest_lum_w_a value.
    z_qso : object
        z_qso value.
    """
    d_l_cm = _luminosity_distance_cm_jax(z_qso)
    flux_cgs_rest = (
        jnp.asarray(host_rest_lum_w_a, dtype=jnp.float64)
        * W_PER_A_TO_CGS_PER_A
        / jnp.maximum(4.0 * jnp.pi * d_l_cm**2, 1.0e-300)
    )
    return flux_cgs_rest * CGS_TO_JAXQSOFIT_FLUX


def _rest_log_lambda_llambda_from_flam(wave_rest, flam_rest, z):
    """Return log10(lambda Llambda) using rest-frame f_lambda in 1e-17 cgs units.

    Parameters
    ----------
    wave_rest : object
        wave_rest value.
    flam_rest : object
        flam_rest value.
    z : object
        z value.
    """
    wave_rest = jnp.maximum(jnp.asarray(wave_rest), 1e-8)
    flam_rest_cgs = 1e-17 * jnp.asarray(flam_rest)
    d_l_cm = _luminosity_distance_cm_jax(z)
    lambda_llambda = 4.0 * jnp.pi * d_l_cm**2 * wave_rest * flam_rest_cgs
    return jnp.log10(jnp.clip(lambda_llambda, 1e-300, None))


def _powerlaw_jax(wave, pl_norm, pl_slope, pivot):
    """Evaluate a power-law continuum at input wavelengths.

    Parameters
    ----------
    wave : object
        wave value.
    pl_norm : object
        pl_norm value.
    pl_slope : object
        pl_slope value.
    pivot : object
        pivot value.
    """
    x = jnp.clip(wave / pivot, 1e-8, None)
    return pl_norm * x ** pl_slope


def _host_redshift_prior_params(prior_config, z_qso):
    """Return smooth redshift-dependent host prior weight, loc shift, scale multiplier, and df.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    z_qso : object
        z_qso value.
    """
    cfg = prior_config.get("host_redshift_prior", {}) if isinstance(prior_config, Mapping) else {}
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
    """Additive negative BAL trough with optional super-Gaussian boxiness.

    Parameters
    ----------
    wave : object
        wave value.
    params : object
        params value.
    metadata : object
        metadata value.
    """
    center = params["center"]
    sigma = jnp.maximum(params["sigma"], 1e-3)
    depth = jnp.maximum(params["depth"], 0.0)
    # ``shape_power=2`` reproduces the legacy Gaussian profile exactly.
    shape_power = jnp.maximum(params.get("shape_power", 2.0), 2.0)
    x = (wave - center) / sigma
    return -depth * jnp.exp(-0.5 * jnp.abs(x) ** shape_power)


def gaussian_bal_optical_depth_component(wave, params, metadata):
    """BAL optical-depth profile parameterized by outflow velocity.

    Parameters
    ----------
    wave : object
        wave value.
    params : object
        params value.
    metadata : object
        metadata value.
    """
    line_lambda = jnp.asarray(metadata["line_lambda"], dtype=jnp.float64)
    v_out = jnp.maximum(params["v_out"], 0.0)
    center = line_lambda * (1.0 - v_out / C_KMS)
    sigma_v_kms = jnp.maximum(params["fwhm_kms"], 1.0) / 2.354820045
    sigma = jnp.maximum(jnp.abs(center) * sigma_v_kms / C_KMS, 1e-3)
    tau_peak = jnp.maximum(params["tau_peak"], 0.0)
    shape_power = jnp.maximum(params.get("shape_power", 2.0), 2.0)
    x = (wave - center) / sigma
    return tau_peak * jnp.exp(-0.5 * jnp.abs(x) ** shape_power)


def _is_multiplicative_bal_component(comp):
    """Return True for built-in BAL components modeled as transmission.


    Parameters
    ----------
    comp : object
        comp value.
    """
    return str(getattr(comp, "metadata", {}).get("component_type", "")) == "bal_absorption"


def _bal_covering_fraction(params):
    """Return a bounded BAL covering fraction.

    Parameters
    ----------
    params : object
        params value.
    """
    return jnp.clip(jnp.asarray(params.get("covering", 1.0)), 0.0, 0.999)


def _smc_like_reddening_jax(wave, a_uv, uv_ref=2500.0, alpha=1.2):
    """Return a smooth SMC-like attenuation curve.

    The amplitude is normalized at ``uv_ref``: ``a_uv`` is
    :math:`A(\\mathrm{uv\\_ref})` in magnitudes, not a literal
    color excess.

    Parameters
    ----------
    wave : object
        wave value.
    a_uv : object
        a_uv value.
    uv_ref : object
        uv_ref value.
    alpha : object
        alpha value.
    """
    a_uv = jnp.maximum(jnp.asarray(a_uv), 0.0)
    uv_ref = jnp.maximum(jnp.asarray(uv_ref), 1e-8)
    alpha = jnp.asarray(alpha)
    k_lambda = (jnp.clip(wave, 1e-8, None) / uv_ref) ** (-alpha)
    return 10.0 ** (-0.4 * a_uv * k_lambda)


def _many_gauss_lnlam(lnlam, amps, mus, sigs):
    """Sum Gaussian components defined in log-wavelength space.

    Parameters
    ----------
    lnlam : object
        lnlam value.
    amps : object
        amps value.
    mus : object
        mus value.
    sigs : object
        sigs value.
    """
    z = (lnlam[:, None] - mus[None, :]) / sigs[None, :]
    return jnp.sum(amps[None, :] * jnp.exp(-0.5 * z * z), axis=1)


def _split_many_gauss_lnlam(lnlam, amps, mus, sigs, broad_mask, *, return_profiles=False):
    """Evaluate tied Gaussian lines once and split them into broad/narrow sums.

    Parameters
    ----------
    lnlam : object
        lnlam value.
    amps : object
        amps value.
    mus : object
        mus value.
    sigs : object
        sigs value.
    broad_mask : object
        broad_mask value.
    return_profiles : object
        return_profiles value.
    """
    profiles_pixel_major = amps[None, :] * jnp.exp(
        -0.5 * ((lnlam[:, None] - mus[None, :]) / sigs[None, :]) ** 2
    )
    total = jnp.sum(profiles_pixel_major, axis=1)
    broad = jnp.sum(profiles_pixel_major * broad_mask[None, :], axis=1)
    narrow = total - broad
    if return_profiles:
        return total, broad, narrow, jnp.swapaxes(profiles_pixel_major, 0, 1)
    return total, broad, narrow, None


def _line_meta_array(meta, key, *, jax_key=None, dtype=jnp.float64):
    """Return JAX-ready line metadata, preferring precomputed static arrays.

    Parameters
    ----------
    meta : object
        meta value.
    key : object
        key value.
    jax_key : object
        jax_key value.
    dtype : object
        dtype value.
    """
    if jax_key is not None and jax_key in meta:
        return meta[jax_key]
    return jnp.asarray(meta[key], dtype=dtype)


def _broad_line_mask(names):
    """Return a float mask identifying broad-line components by name.

    Parameters
    ----------
    names : object
        names value.
    """
    return np.asarray(
        [str(name).lower().endswith('_br') or ('_br' in str(name).lower()) for name in names],
        dtype=np.float64,
    )


def _prefixed_site(prefix: str, name: str) -> str:
    """Return a site name with an optional component prefix.


    Parameters
    ----------
    prefix : object
        prefix value.
    name : object
        name value.
    """
    return f"{prefix}_{name}" if prefix else name


def _smooth_bounded_affine(eps, loc, scale, low, high):
    """Map standardized coordinates into bounded space without flat clipping.

    Parameters
    ----------
    eps : object
        eps value.
    loc : object
        loc value.
    scale : object
        scale value.
    low : object
        low value.
    high : object
        high value.
    """
    loc = jnp.asarray(loc, dtype=jnp.float64)
    scale = jnp.maximum(jnp.asarray(scale, dtype=jnp.float64), 1.0e-12)
    low = jnp.asarray(low, dtype=jnp.float64)
    high = jnp.asarray(high, dtype=jnp.float64)
    eps = jnp.asarray(eps, dtype=jnp.float64)

    finite_bounds = jnp.isfinite(low) & jnp.isfinite(high)
    raw_affine = loc + scale * eps

    safe_low = jnp.where(jnp.isfinite(low), low, loc - 1.0)
    safe_high = jnp.where(jnp.isfinite(high), high, loc + 1.0)
    span = jnp.maximum(safe_high - safe_low, 1.0e-12)
    unit_loc = jnp.clip((loc - safe_low) / span, 1.0e-6, 1.0 - 1.0e-6)
    logit_loc = jnp.log(unit_loc) - jnp.log1p(-unit_loc)
    local_slope = jnp.maximum(span * unit_loc * (1.0 - unit_loc), 1.0e-12)
    raw_bounded = logit_loc + (scale / local_slope) * eps
    bounded = safe_low + span * jax.nn.sigmoid(raw_bounded)
    return jnp.where(finite_bounds, bounded, raw_affine)


def _sample_bounded_affine_std(site_name, loc, scale, low, high):
    """Sample standardized coordinates and transform smoothly into bounds.

    Parameters
    ----------
    site_name : object
        site_name value.
    loc : object
        loc value.
    scale : object
        scale value.
    low : object
        low value.
    high : object
        high value.
    """
    loc = jnp.asarray(loc, dtype=jnp.float64)
    scale = jnp.maximum(jnp.asarray(scale, dtype=jnp.float64), 1.0e-12)
    low = jnp.asarray(low, dtype=jnp.float64)
    high = jnp.asarray(high, dtype=jnp.float64)
    eps_dist = dist.Normal(jnp.zeros_like(loc), jnp.ones_like(loc))
    if int(jnp.ndim(loc)) > 0:
        eps_dist = eps_dist.to_event(1)
    eps = numpyro.sample(f"{site_name}_std", eps_dist)
    value = _smooth_bounded_affine(eps, loc, scale, low, high)
    return numpyro.deterministic(site_name, value)


def _sample_tied_line_groups(tied_line_meta, prior_config, *, site_prefix: str = ""):
    """Sample tied-line groups in geometry-friendly coordinates.

    Velocity shifts use non-centered standardized offsets. Widths are sampled as
    broad/narrow log-FWHM family scales plus per-width-group log offsets. Line
    amplitudes use the direct bounded peak-amplitude prior. The returned arrays
    preserve the historical physical group names.

    Parameters
    ----------
    tied_line_meta : object
        tied_line_meta value.
    prior_config : object
        prior_config value.
    site_prefix : object
        site_prefix value.
    """
    n_v = int(tied_line_meta["n_vgroups"])
    n_w = int(tied_line_meta["n_wgroups"])
    n_f = int(tied_line_meta["n_fgroups"])
    dmu_scale_mult = float(prior_config["line_dmu_scale_mult"])
    sig_scale_mult = float(prior_config["line_sig_scale_mult"])
    amp_scale_mult = float(prior_config["line_amp_scale_mult"])

    dmu_group = jnp.zeros((0,), dtype=jnp.float64)
    if n_v > 0:
        dmu_min = _line_meta_array(tied_line_meta, "dmu_min_group", jax_key="dmu_min_group_jax")
        dmu_max = _line_meta_array(tied_line_meta, "dmu_max_group", jax_key="dmu_max_group_jax")
        dmu_group = _sample_bounded_affine_std(
            _prefixed_site(site_prefix, "line_dmu_group"),
            _line_meta_array(tied_line_meta, "dmu_init_group", jax_key="dmu_init_group_jax"),
            jnp.maximum(dmu_scale_mult * (dmu_max - dmu_min), 1.0e-6),
            dmu_min,
            dmu_max,
        )

    sig_group = jnp.zeros((0,), dtype=jnp.float64)
    if n_w > 0:
        sig_init = np.clip(np.asarray(tied_line_meta["sig_init_group"], dtype=float), 1.0e-8, None)
        sig_min = np.clip(np.asarray(tied_line_meta["sig_min_group"], dtype=float), 1.0e-8, None)
        sig_max = np.clip(np.asarray(tied_line_meta["sig_max_group"], dtype=float), 1.0e-8, None)
        log_fwhm_init = np.log(C_KMS * 2.354820045 * sig_init)
        log_fwhm_min = np.log(C_KMS * 2.354820045 * sig_min)
        log_fwhm_max = np.log(C_KMS * 2.354820045 * sig_max)
        wgroup = np.asarray(tied_line_meta["wgroup"], dtype=int)
        broad_mask = np.asarray(tied_line_meta.get("broad_mask", _broad_line_mask(tied_line_meta.get("names", []))), dtype=float)
        wgroup_is_broad = np.asarray(
            [np.any(broad_mask[wgroup == gid] > 0.0) for gid in range(n_w)],
            dtype=bool,
        )
        broad_idx = np.where(wgroup_is_broad)[0]
        narrow_idx = np.where(~wgroup_is_broad)[0]
        broad_default = float(np.median(log_fwhm_init[broad_idx])) if broad_idx.size else np.log(3000.0)
        narrow_default = float(np.median(log_fwhm_init[narrow_idx])) if narrow_idx.size else np.log(500.0)

        log_broad_fwhm = (
            _sample_bounded_affine_std(
                _prefixed_site(site_prefix, "line_log_broad_fwhm"),
                broad_default,
                0.35,
                float(np.min(log_fwhm_min[broad_idx])) if broad_idx.size else -jnp.inf,
                float(np.max(log_fwhm_max[broad_idx])) if broad_idx.size else jnp.inf,
            )
            if broad_idx.size
            else jnp.asarray(broad_default, dtype=jnp.float64)
        )
        log_narrow_fwhm = (
            _sample_bounded_affine_std(
                _prefixed_site(site_prefix, "line_log_narrow_fwhm"),
                narrow_default,
                0.25,
                float(np.min(log_fwhm_min[narrow_idx])) if narrow_idx.size else -jnp.inf,
                float(np.max(log_fwhm_max[narrow_idx])) if narrow_idx.size else jnp.inf,
            )
            if narrow_idx.size
            else jnp.asarray(narrow_default, dtype=jnp.float64)
        )
        family_loc = np.where(wgroup_is_broad, broad_default, narrow_default)
        family_base = jnp.where(
            jnp.asarray(wgroup_is_broad, dtype=bool),
            log_broad_fwhm,
            log_narrow_fwhm,
        )
        delta_loc = jnp.asarray(log_fwhm_init - family_loc, dtype=jnp.float64)
        delta_scale = jnp.maximum(
            sig_scale_mult * jnp.asarray(log_fwhm_max - log_fwhm_min, dtype=jnp.float64),
            1.0e-4,
        )
        delta = _sample_bounded_affine_std(
            _prefixed_site(site_prefix, "line_log_fwhm_delta_group"),
            delta_loc,
            delta_scale,
            jnp.asarray(log_fwhm_min, dtype=jnp.float64) - family_base,
            jnp.asarray(log_fwhm_max, dtype=jnp.float64) - family_base,
        )
        log_fwhm_group = family_base + delta
        numpyro.deterministic(_prefixed_site(site_prefix, "line_log_fwhm_group"), log_fwhm_group)
        sig_group = numpyro.deterministic(
            _prefixed_site(site_prefix, "line_sig_group"),
            jnp.exp(log_fwhm_group) / (C_KMS * 2.354820045),
        )

    amp_group = jnp.zeros((0,), dtype=jnp.float64)
    if n_f > 0:
        amp_min = jnp.clip(
            _line_meta_array(tied_line_meta, "amp_min_group", jax_key="amp_min_group_jax"),
            AMPLITUDE_FLOOR,
        )
        amp_max = jnp.clip(
            _line_meta_array(tied_line_meta, "amp_max_group", jax_key="amp_max_group_jax"),
            AMPLITUDE_FLOOR,
        )
        amp_init = jnp.clip(
            _line_meta_array(tied_line_meta, "amp_init_group", jax_key="amp_init_group_jax"),
            AMPLITUDE_FLOOR,
        )
        amp_group = numpyro.sample(
            _prefixed_site(site_prefix, "line_amp_group"),
            dist.TruncatedNormal(
                loc=amp_init,
                scale=jnp.maximum(amp_scale_mult * (amp_max - amp_min), AMPLITUDE_FLOOR),
                low=amp_min,
                high=amp_max,
            ),
        )

    return dmu_group, sig_group, amp_group


def _synth_ab_mag_from_grid(wave_obs, flam_obs, filt_trans):
    """Compute an AB magnitude from flux density and filter transmission on one grid.

    Parameters
    ----------
    wave_obs : object
        wave_obs value.
    flam_obs : object
        flam_obs value.
    filt_trans : object
        filt_trans value.
    """
    c_ang_s = 2.99792458e18
    trans = jnp.clip(filt_trans, 0.0, None)
    # Model spectra are stored in SDSS-style 1e-17 flux-density units.
    flam_obs_cgs = 1e-17 * flam_obs
    num = jnp.trapezoid(flam_obs_cgs * trans * wave_obs, wave_obs)
    den = jnp.trapezoid(trans * c_ang_s / jnp.clip(wave_obs, 1e-8, None), wave_obs)
    fnu = num / jnp.maximum(den, 1e-30)
    return -2.5 * jnp.log10(jnp.clip(fnu, 1e-30, None)) - 48.60


def _shift_and_broaden_single_spectrum_lnlam(lnwave, spectrum, v_kms, sigma_kms, *, convolution_method="fft"):
    """Apply LOS velocity shift and Gaussian broadening to one spectrum.

    Parameters
    ----------
    lnwave : object
        lnwave value.
    spectrum : object
        spectrum value.
    v_kms : object
        v_kms value.
    sigma_kms : object
        sigma_kms value.
    convolution_method : object
        convolution_method value.
    """
    sigma_ln = jnp.maximum(sigma_kms / C_KMS, 1e-5)

    wave = jnp.exp(lnwave)
    shift_ln = v_kms / C_KMS
    shifted_wave = jnp.exp(lnwave - shift_ln)
    shifted = jnp.interp(shifted_wave, wave, spectrum, left=0.0, right=0.0)
    return _convolve_velocity_space(
        lnwave,
        shifted,
        sigma_ln,
        radius_mult=5.0,
        max_half=512,
        method=convolution_method,
    )


def _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=512):
    """Build a normalized 1D Gaussian convolution kernel with fixed max size.

    Parameters
    ----------
    sigma_pix : object
        sigma_pix value.
    radius_mult : object
        radius_mult value.
    max_half : object
        max_half value.
    """
    sigma_pix = jnp.maximum(sigma_pix, 1e-3)
    x = jnp.arange(-max_half, max_half + 1, dtype=jnp.float64)
    half_dyn = jnp.maximum(3.0, jnp.ceil(radius_mult * sigma_pix))
    mask = jnp.abs(x) <= half_dyn
    k = jnp.exp(-0.5 * (x / sigma_pix) ** 2)
    k = jnp.where(mask, k, 0.0)
    return k / jnp.maximum(jnp.sum(k), 1e-30)


def _convolve_same_length_direct(signal, kernel):
    """Convolve and center-crop so the output matches the signal length.

    Parameters
    ----------
    signal : object
        signal value.
    kernel : object
        kernel value.
    """
    signal = jnp.asarray(signal)
    kernel = jnp.asarray(kernel)
    full = jnp.convolve(signal, kernel, mode='same')
    n = signal.shape[0]
    m = full.shape[0]
    start = jnp.maximum((m - n) // 2, 0)
    return jax.lax.dynamic_slice(full, (start,), (n,))


def _convolve_same_length_fft(signal, kernel):
    """Linear FFT convolution with the same centered crop as the direct path.

    Parameters
    ----------
    signal : object
        signal value.
    kernel : object
        kernel value.
    """
    signal = jnp.asarray(signal)
    kernel = jnp.asarray(kernel)
    n = signal.shape[0]
    k = kernel.shape[0]
    full_len = n + k - 1
    signal_fft = jnp.fft.rfft(signal, n=full_len)
    kernel_fft = jnp.fft.rfft(kernel, n=full_len)
    full = jnp.fft.irfft(signal_fft * kernel_fft, n=full_len)
    start = (k - 1) // 2
    return jax.lax.dynamic_slice(full, (start,), (n,))


def _convolve_same_length(signal, kernel, *, method="fft"):
    """Convolve with a selectable JAX backend and direct fallback for short arrays.

    Parameters
    ----------
    signal : object
        signal value.
    kernel : object
        kernel value.
    method : object
        method value.
    """
    signal = jnp.asarray(signal)
    kernel = jnp.asarray(kernel)
    method = str(method).lower()
    if method == "direct":
        return _convolve_same_length_direct(signal, kernel)
    if method != "fft":
        raise ValueError("convolution method must be 'fft' or 'direct'.")
    if signal.shape[0] <= kernel.shape[0]:
        return _convolve_same_length_direct(signal, kernel)
    return _convolve_same_length_fft(signal, kernel)


def _convolve_velocity_space(lnwave, signal, sigma_ln, radius_mult=5.0, max_half=512, *, method="fft"):
    """Convolve a spectrum with a Gaussian of fixed width in log-wavelength.

    The input grid may be linear, logarithmic, or otherwise monotonic. The
    convolution is performed on an internal uniform log-wavelength grid and then
    interpolated back to the requested grid.

    Parameters
    ----------
    lnwave : object
        lnwave value.
    signal : object
        signal value.
    sigma_ln : object
        sigma_ln value.
    radius_mult : object
        radius_mult value.
    max_half : object
        max_half value.
    method : object
        method value.
    """
    lnwave = jnp.asarray(lnwave, dtype=jnp.float64)
    signal = jnp.asarray(signal, dtype=jnp.float64)
    n = lnwave.shape[0]
    ln_uniform = jnp.linspace(lnwave[0], lnwave[-1], n)
    dln = jnp.maximum((lnwave[-1] - lnwave[0]) / jnp.maximum(n - 1, 1), 1e-8)
    sigma_pix = jnp.maximum(sigma_ln, 1e-8) / dln
    kern = _gaussian_kernel1d(sigma_pix, radius_mult=radius_mult, max_half=max_half)
    signal_uniform = jnp.interp(ln_uniform, lnwave, signal, left=0.0, right=0.0)
    convolved_uniform = _convolve_same_length(signal_uniform, kern, method=method)
    return jnp.interp(lnwave, ln_uniform, convolved_uniform, left=0.0, right=0.0)


def _fe_template_component(
    wave,
    wave_template,
    flux_template,
    norm,
    fwhm_kms,
    shift_frac,
    base_fwhm_kms=900.0,
    *,
    template_on_wave=None,
    convolution_method="fft",
):
    """Generate a broadened and shifted Fe template contribution.

    Parameters
    ----------
    wave : object
        wave value.
    wave_template : object
        wave_template value.
    flux_template : object
        flux_template value.
    norm : object
        norm value.
    fwhm_kms : object
        fwhm_kms value.
    shift_frac : object
        shift_frac value.
    base_fwhm_kms : object
        base_fwhm_kms value.
    template_on_wave : object
        template_on_wave value.
    convolution_method : object
        convolution_method value.
    """
    # Enforce physically non-negative Fe pseudo-continuum and model broadening in velocity space.
    if template_on_wave is None:
        flux_template = jnp.maximum(flux_template, 0.0)
        template_on_wave = jnp.interp(wave, wave_template, flux_template, left=0.0, right=0.0)
    else:
        template_on_wave = jnp.maximum(jnp.asarray(template_on_wave, dtype=jnp.float64), 0.0)

    min_fwhm_kms = 1.01 * base_fwhm_kms
    transition_kms = jnp.maximum(0.01 * base_fwhm_kms, 10.0)
    fwhm_total = min_fwhm_kms + transition_kms * jax.nn.softplus((fwhm_kms - min_fwhm_kms) / transition_kms)
    fwhm_eff = jnp.sqrt(fwhm_total**2 - base_fwhm_kms**2)
    sigma_kms = fwhm_eff / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    v_kms = C_KMS * shift_frac
    lnwave = jnp.log(wave)
    model = _shift_and_broaden_single_spectrum_lnlam(
        lnwave,
        template_on_wave,
        v_kms,
        sigma_kms,
        convolution_method=convolution_method,
    )
    return norm * model


def _balmer_static_terms_jax(wave, balmer_te=15000.0):
    """Return wavelength-only Balmer continuum terms for a fixed electron temperature.

    Parameters
    ----------
    wave : object
        wave value.
    balmer_te : object
        balmer_te value.
    """
    lam_be = 3646.0
    h = 6.62607015e-27
    c = 2.99792458e10
    kb = 1.380649e-16

    wave = jnp.asarray(wave)
    lam_cm = wave * 1e-8
    balmer_te = jnp.asarray(balmer_te, dtype=jnp.float64)

    expo = jnp.clip((h * c) / (lam_cm * kb * balmer_te), 1e-9, 700.0)
    bb = (2.0 * h * c**2 / lam_cm**5) / jnp.expm1(expo)
    bb = bb * 1e-8 * jnp.pi

    # Normalize shape at Balmer edge so balmer_norm is a flux-like amplitude.
    lam_be_cm = lam_be * 1e-8
    expo_edge = jnp.clip((h * c) / (lam_be_cm * kb * balmer_te), 1e-9, 700.0)
    bb_edge = (2.0 * h * c**2 / lam_be_cm**5) / jnp.expm1(expo_edge)
    bb_edge = bb_edge * 1e-8 * jnp.pi
    bb = bb / jnp.maximum(bb_edge, 1e-30)
    tau_shape = (wave / lam_be) ** 3
    below_edge = wave <= lam_be
    return bb, tau_shape, below_edge


def _balmer_continuum_jax(
    wave,
    balmer_norm,
    balmer_te,
    balmer_tau,
    balmer_vel,
    *,
    balmer_static_terms=None,
    convolution_method="fft",
):
    """Compute Balmer continuum template with edge-normalized blackbody shape.

    Parameters
    ----------
    wave : object
        wave value.
    balmer_norm : object
        balmer_norm value.
    balmer_te : object
        balmer_te value.
    balmer_tau : object
        balmer_tau value.
    balmer_vel : object
        balmer_vel value.
    balmer_static_terms : object
        balmer_static_terms value.
    convolution_method : object
        convolution_method value.
    """
    if balmer_static_terms is None:
        bb, tau_shape, below_edge = _balmer_static_terms_jax(wave, balmer_te=balmer_te)
    else:
        bb, tau_shape, below_edge = balmer_static_terms
        bb = jnp.asarray(bb, dtype=jnp.float64)
        tau_shape = jnp.asarray(tau_shape, dtype=jnp.float64)
        below_edge = jnp.asarray(below_edge, dtype=bool)

    tau = balmer_tau * tau_shape
    bc = balmer_norm * (1.0 - jnp.exp(-tau)) * bb
    bc = jnp.where(below_edge, bc, 0.0)

    lnwave = jnp.log(wave)
    sigma_ln = jnp.maximum(balmer_vel / C_KMS, 1e-5)
    bc_conv = _convolve_velocity_space(lnwave, bc, sigma_ln, method=convolution_method)
    return bc_conv


def _prior_distribution(prior_config, key, default_distribution):
    """Read a NumPyro distribution-like prior from a flat prior mapping.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    key : object
        key value.
    default_distribution : object
        default_distribution value.
    """
    cfg = prior_config.get(key, None)
    if cfg is None:
        return default_distribution
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return dist.Normal(
            jnp.asarray(cfg[0], dtype=jnp.float64),
            jnp.maximum(jnp.asarray(cfg[1], dtype=jnp.float64), 1.0e-6),
        )
    if not isinstance(cfg, Mapping):
        return default_distribution

    dist_name = str(cfg.get("dist", cfg.get("family", default_distribution.__class__.__name__))).lower()
    if dist_name in {"delta", "fixed", "deterministic"}:
        return None
    if dist_name in {"normal", "gaussian"}:
        return dist.Normal(
            jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64),
            jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6),
        )
    if dist_name in {"truncatednormal", "truncated_normal", "truncnormal", "truncnorm"}:
        return dist.TruncatedNormal(
            loc=jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64),
            scale=jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6),
            low=jnp.asarray(cfg.get("low", -jnp.inf), dtype=jnp.float64),
            high=jnp.asarray(cfg.get("high", jnp.inf), dtype=jnp.float64),
        )
    if dist_name in {"lognormal", "log-normal", "log_normal"}:
        return dist.LogNormal(
            jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64),
            jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6),
        )
    if dist_name in {"halfnormal", "half_normal"}:
        return dist.HalfNormal(jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6))
    if dist_name in {"student_t", "studentt", "studentt", "t"}:
        return dist.StudentT(
            df=jnp.maximum(jnp.asarray(cfg.get("df", 5.0), dtype=jnp.float64), 1.0e-6),
            loc=jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64),
            scale=jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6),
        )
    if dist_name in {"uniform", "flat"}:
        low = jnp.asarray(cfg.get("low", 0.0), dtype=jnp.float64)
        high = jnp.asarray(cfg.get("high", 1.0), dtype=jnp.float64)
        lo = jnp.minimum(low, high)
        hi = jnp.maximum(jnp.maximum(low, high), lo + 1.0e-6)
        return dist.Uniform(lo, hi)
    if dist_name in {"exponential", "exp"}:
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-30)
        return dist.Exponential(1.0 / scale)
    return default_distribution


def _fixed_prior_value(prior_config, key, default_value):
    """_fixed_prior_value helper.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    key : object
        key value.
    default_value : object
        default_value value.
    """
    cfg = prior_config.get(key, None)
    if isinstance(cfg, Mapping):
        dist_name = str(cfg.get("dist", cfg.get("family", ""))).lower()
        if dist_name in {"delta", "fixed", "deterministic"}:
            return jnp.asarray(cfg.get("value", cfg.get("loc", default_value)), dtype=jnp.float64)
    return None


def _prior_family(prior_config, key, default_family=""):
    """_prior_family helper.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    key : object
        key value.
    default_family : object
        default_family value.
    """
    cfg = prior_config.get(key, None)
    if isinstance(cfg, Mapping):
        return str(cfg.get("dist", cfg.get("family", default_family))).lower()
    return default_family.lower()


def _prior_loc_scale(prior_config, key, default_loc=0.0, default_scale=1.0):
    """_prior_loc_scale helper.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    key : object
        key value.
    default_loc : object
        default_loc value.
    default_scale : object
        default_scale value.
    """
    cfg = prior_config.get(key, None)
    if isinstance(cfg, Mapping):
        return (
            jnp.asarray(cfg.get("loc", default_loc), dtype=jnp.float64),
            jnp.maximum(jnp.asarray(cfg.get("scale", default_scale), dtype=jnp.float64), 1.0e-6),
        )
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return (
            jnp.asarray(cfg[0], dtype=jnp.float64),
            jnp.maximum(jnp.asarray(cfg[1], dtype=jnp.float64), 1.0e-6),
        )
    return jnp.asarray(default_loc, dtype=jnp.float64), jnp.asarray(default_scale, dtype=jnp.float64)


def _halfnormal_prior(prior_config, key, default_scale, *, ref_scale=None):
    """_halfnormal_prior helper.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    key : object
        key value.
    default_scale : object
        default_scale value.
    ref_scale : object
        ref_scale value.
    """
    cfg = prior_config.get(key, None)
    if isinstance(cfg, Mapping) and "scale_mult_err" in cfg and ref_scale is not None:
        return dist.HalfNormal(jnp.maximum(jnp.asarray(cfg["scale_mult_err"] * ref_scale, dtype=jnp.float64), 1.0e-6))
    return _prior_distribution(prior_config, key, dist.HalfNormal(default_scale))


def _sample_prior(prior_config, key, default_distribution):
    """Sample a scalar site from a configured distribution or a default.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    key : object
        key value.
    default_distribution : object
        default_distribution value.
    """
    fixed = _fixed_prior_value(prior_config, key, None)
    if fixed is not None:
        return numpyro.deterministic(key, fixed)
    return numpyro.sample(key, _prior_distribution(prior_config, key, default_distribution))


def _sample_log_positive_from_distribution(prior_config, *, value_key, log_key, default_distribution):
    """Sample a log-parameter from a distribution and expose its physical value.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    value_key : object
        value_key value.
    log_key : object
        log_key value.
    default_distribution : object
        default_distribution value.
    """
    fixed = _fixed_prior_value(prior_config, log_key, None)
    if fixed is not None:
        log_value = numpyro.deterministic(log_key, fixed)
    else:
        log_value = numpyro.sample(log_key, _prior_distribution(prior_config, log_key, default_distribution))
    return numpyro.deterministic(value_key, jnp.exp(log_value))


def _sample_positive_distribution(
    prior_config,
    *,
    value_key,
    log_key,
    default_value_distribution,
    default_log_distribution,
    default_to_log=False,
):
    """Sample a positive parameter, honoring either physical or log prior keys.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    value_key : object
        value_key value.
    log_key : object
        log_key value.
    default_value_distribution : object
        default_value_distribution value.
    default_log_distribution : object
        default_log_distribution value.
    default_to_log : object
        default_to_log value.
    """
    if log_key in prior_config:
        family = _prior_family(prior_config, log_key, default_log_distribution.__class__.__name__)
        if family in {"lognormal", "log-normal", "log_normal", "exponential", "exp", "halfnormal", "half_normal"}:
            fixed = _fixed_prior_value(prior_config, log_key, None)
            if fixed is not None:
                return numpyro.deterministic(value_key, fixed)
            return numpyro.sample(value_key, _prior_distribution(prior_config, log_key, default_value_distribution))
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=default_log_distribution,
        )
    fixed = _fixed_prior_value(prior_config, value_key, None)
    if fixed is not None:
        return numpyro.deterministic(value_key, fixed)
    if value_key not in prior_config and default_to_log:
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=default_log_distribution,
        )
    return numpyro.sample(value_key, _prior_distribution(prior_config, value_key, default_value_distribution))


def _template_grid_age_met_arrays(fsps_grid):
    """Return flattened age and metallicity arrays matching template order.

    Parameters
    ----------
    fsps_grid : object
        fsps_grid value.
    """
    ages = np.asarray([m.get("tage_gyr", np.nan) for m in fsps_grid.template_meta], dtype=float)
    mets = np.asarray([m.get("logzsol", np.nan) for m in fsps_grid.template_meta], dtype=float)
    if ages.size != fsps_grid.templates.shape[1] or not np.all(np.isfinite(ages)):
        ages = np.tile(np.asarray(fsps_grid.age_grid_gyr, dtype=float), len(fsps_grid.logzsol_grid))
    if mets.size != fsps_grid.templates.shape[1] or not np.all(np.isfinite(mets)):
        mets = np.repeat(np.asarray(fsps_grid.logzsol_grid, dtype=float), len(fsps_grid.age_grid_gyr))
    return _np_to_jnp(ages), _np_to_jnp(mets)


def _flexible_host_raw_weight_locs(fsps_grid, prior_config, ntemp):
    """Return template-wise prior logits for flexible host SSP weights.

    Parameters
    ----------
    fsps_grid : object
        fsps_grid value.
    prior_config : object
        prior_config value.
    ntemp : object
        ntemp value.
    """
    raw_w_loc, _ = _prior_loc_scale(prior_config, "raw_w", -0.5, 1.0)
    loc = jnp.full((ntemp,), raw_w_loc, dtype=jnp.float64)
    cfg = prior_config.get("host_template_age_prior", None)
    if cfg is None:
        cfg = prior_config.get("template_age_prior", None)
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", True)):
        return loc

    ages = np.asarray([m.get("tage_gyr", np.nan) for m in fsps_grid.template_meta], dtype=float)
    if ages.size != ntemp or not np.all(np.isfinite(ages)):
        ages = np.asarray(_template_grid_age_met_arrays(fsps_grid)[0], dtype=float)
    if ages.size != ntemp:
        return loc

    pivot_gyr = max(float(cfg.get("pivot_gyr", 1.0)), 1.0e-6)
    strength = float(cfg.get("strength", 1.0))
    min_logit = float(cfg.get("min_logit", -3.0))
    max_logit = float(cfg.get("max_logit", 2.0))
    safe_ages = np.where(np.isfinite(ages) & (ages > 0.0), ages, pivot_gyr)
    prior_type = str(cfg.get("type", "prefer_old")).lower()

    if prior_type in {"prefer_old", "old", "older", "old_host"}:
        age_logits = strength * np.log10(np.maximum(safe_ages, 1.0e-6) / pivot_gyr)
    elif prior_type in {"lognormal", "log_age_normal", "age_peak"}:
        loc_gyr = max(float(cfg.get("loc_gyr", pivot_gyr)), 1.0e-6)
        scale_dex = max(float(cfg.get("scale_dex", 0.5)), 1.0e-6)
        log_age = np.log10(np.maximum(safe_ages, 1.0e-6))
        age_logits = -0.5 * strength * ((log_age - np.log10(loc_gyr)) / scale_dex) ** 2
    else:
        raise ValueError("host_template_age_prior type must be 'prefer_old' or 'lognormal'.")

    age_logits = np.clip(age_logits, min_logit, max_logit)
    return loc + jnp.asarray(age_logits, dtype=jnp.float64)


def _proxy_template_weights_from_host_state(fsps_grid, host_state):
    """Map full JAXSEDFit SSP weights onto the legacy template grid for summaries.

    Parameters
    ----------
    fsps_grid : object
        fsps_grid value.
    host_state : object
        host_state value.
    """
    template_age_gyr, template_lgmet = _template_grid_age_met_arrays(fsps_grid)
    meta_lg_age = np.asarray(
        [
            m.get("dsps_lg_age_gyr", np.log10(max(m.get("tage_gyr", 1e-5), 1e-5)))
            for m in fsps_grid.template_meta
        ],
        dtype=float,
    )
    meta_lgmet = np.asarray(
        [m.get("dsps_lgmet", m.get("logzsol", 0.0)) for m in fsps_grid.template_meta],
        dtype=float,
    )
    if meta_lg_age.size != fsps_grid.templates.shape[1] or not np.all(np.isfinite(meta_lg_age)):
        meta_lg_age = np.log10(np.maximum(np.asarray(template_age_gyr, dtype=float), 1e-5))
    if meta_lgmet.size != fsps_grid.templates.shape[1] or not np.all(np.isfinite(meta_lgmet)):
        meta_lgmet = np.asarray(template_lgmet, dtype=float)

    ssp_lg_age = np.asarray(host_state["ssp_lg_age_gyr"], dtype=float)
    ssp_lgmet = np.asarray(host_state["ssp_lgmet"], dtype=float)
    age_idx = np.asarray([int(np.argmin(np.abs(ssp_lg_age - x))) for x in meta_lg_age], dtype=int)
    met_idx = np.asarray([int(np.argmin(np.abs(ssp_lgmet - x))) for x in meta_lgmet], dtype=int)
    weights_frac = host_state["host_ssp_weights"][met_idx, age_idx]
    return weights_frac / jnp.maximum(jnp.sum(weights_frac), 1e-30)


def _sample_log_host_aperture_scale(prior_config):
    """Return log aperture scale for the physical host spectrum.

    ``log_host_aperture_scale`` multiplies the host luminosity-derived spectrum
    after conversion to flux. The default deterministic value is 0, i.e.
    aperture scale 1, which assumes the fitted spectrum captures the whole
    galaxy light. Override this prior for fiber/slit spectra or known aperture
    losses.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    """
    if "log_host_aperture_scale" not in prior_config:
        return numpyro.deterministic("log_host_aperture_scale", jnp.asarray(0.0, dtype=jnp.float64))
    return _sample_prior(prior_config, "log_host_aperture_scale", dist.Normal(0.0, 1.0))


def _delayed_sfh_template_weights_compat(fsps_grid, prior_config, host_amp):
    """Compatibility delayed-tau path for tests and legacy grids without host_basis_jax.

    Parameters
    ----------
    fsps_grid : object
        fsps_grid value.
    prior_config : object
        prior_config value.
    host_amp : object
        host_amp value.
    """
    template_age_gyr, template_lgmet = _template_grid_age_met_arrays(fsps_grid)
    templates = jnp.asarray(fsps_grid.templates.T, dtype=jnp.float64)

    met_values = np.unique(np.asarray(template_lgmet, dtype=float))
    age_values = np.unique(np.asarray(template_age_gyr, dtype=float))
    while met_values.size < 3:
        met_values = np.append(met_values, met_values[-1] + 0.5)
    while age_values.size < 3:
        age_values = np.append(age_values, age_values[-1] * 1.5 + 0.05)
    met_values = np.asarray(met_values, dtype=float)
    age_values = np.asarray(age_values, dtype=float)
    n_met = int(met_values.size)
    n_age = int(age_values.size)
    met_index = np.searchsorted(met_values, np.asarray(template_lgmet, dtype=float))
    age_index = np.searchsorted(age_values, np.asarray(template_age_gyr, dtype=float))

    rest_llambda_np = np.zeros((n_met, n_age, templates.shape[1]), dtype=float)
    rest_llambda_np[met_index, age_index, :] = np.asarray(templates, dtype=float)
    host_basis = HostBasisJax(
        ssp_lgmet=jnp.asarray(met_values, dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.log10(jnp.maximum(jnp.asarray(age_values, dtype=jnp.float64), 1e-5)),
        rest_llambda=jnp.asarray(rest_llambda_np, dtype=jnp.float64),
        surviving_frac_by_age=jnp.ones((n_age,), dtype=jnp.float64),
        n_ly_per_msun=jnp.zeros((n_met, n_age), dtype=jnp.float64),
        ly_lum_per_msun=jnp.zeros((n_met, n_age), dtype=jnp.float64),
        gal_t_table=jnp.geomspace(
            jnp.asarray(0.01, dtype=jnp.float64),
            jnp.maximum(jnp.asarray(float(np.nanmax(age_values))), jnp.asarray(0.011, dtype=jnp.float64)),
            max(16, n_age),
        ),
    )
    host_state = build_jaxsedfit_host_state(
        host_basis,
        prior_config,
        host_sfh_model="delayed",
        t_obs_gyr=float(np.nanmax(age_values)),
        redshift=float(prior_config.get("z_qso", 0.0)),
    )
    host_weights_grid = host_state["host_ssp_weights"]
    weights_frac = host_weights_grid[met_index, age_index]
    weights_frac = weights_frac / jnp.maximum(jnp.sum(weights_frac), 1e-30)

    numpyro.deterministic("sfh_age_gyr", host_state["sfh_age_gyr"])
    numpyro.deterministic("sfh_tau_gyr", host_state["sfh_tau_gyr"])
    numpyro.deterministic("formed_stellar_mass", host_state["formed_mass"])
    numpyro.deterministic("surviving_mass_fraction", host_state["surviving_mass_fraction"])
    numpyro.deterministic("mass_metallicity_relation_logprior", host_state["mass_metallicity_relation_logprior"])
    return host_amp * weights_frac, weights_frac


def _delayed_sfh_host_spectrum(fsps_grid, prior_config, host_amp, z_qso):
    """Return total delayed-SFH host spectrum, weights, and proxy weights.

    Parameters
    ----------
    fsps_grid : object
        fsps_grid value.
    prior_config : object
        prior_config value.
    host_amp : object
        host_amp value.
    z_qso : object
        z_qso value.
    """
    host_basis = getattr(fsps_grid, "host_basis_jax", None)
    if host_basis is None:
        fsps_weights, fsps_weights_frac = _delayed_sfh_template_weights_compat(fsps_grid, prior_config, host_amp)
        gal_intrinsic = jnp.dot(jnp.asarray(fsps_grid.templates, dtype=jnp.float64), fsps_weights)
        return gal_intrinsic, fsps_weights, fsps_weights_frac

    t_obs_gyr = getattr(fsps_grid, "t_obs_gyr", None)
    if t_obs_gyr is None:
        t_obs_gyr = float(np.nanmax(np.power(10.0, np.asarray(host_basis.ssp_lg_age_gyr, dtype=float))))
    static_redshift = float(prior_config.get("z_qso", 0.0))
    host_state = build_jaxsedfit_host_state(
        host_basis,
        prior_config,
        host_sfh_model="delayed",
        t_obs_gyr=float(t_obs_gyr),
        redshift=static_redshift,
    )
    gal_intrinsic = _host_luminosity_w_a_to_rest_flux_units(host_state["host_rest"], z_qso)
    fsps_weights_frac = _proxy_template_weights_from_host_state(fsps_grid, host_state)
    fsps_weights = fsps_weights_frac

    numpyro.deterministic("sfh_age_gyr", host_state["sfh_age_gyr"])
    numpyro.deterministic("sfh_tau_gyr", host_state["sfh_tau_gyr"])
    numpyro.deterministic("formed_stellar_mass", host_state["formed_mass"])
    numpyro.deterministic("surviving_mass_fraction", host_state["surviving_mass_fraction"])
    numpyro.deterministic("mass_metallicity_relation_logprior", host_state["mass_metallicity_relation_logprior"])
    return gal_intrinsic, fsps_weights, fsps_weights_frac


def _sample_from_prior_config(key, cfg):
    """Sample one parameter from a lightweight prior config dictionary.

    Parameters
    ----------
    key : object
        key value.
    cfg : object
        cfg value.
    """
    if isinstance(cfg, Mapping):
        return _sample_prior({key: cfg}, key, dist.Normal(0.0, 1.0))
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return _sample_prior({key: cfg}, key, dist.Normal(0.0, 1.0))
    return numpyro.deterministic(key, jnp.asarray(cfg, dtype=jnp.float64))


def _evaluate_custom_component_jax(wave, samples_or_values, comp, sample_value):
    """Evaluate one custom component from a sample/value mapping.

    Parameters
    ----------
    wave : object
        wave value.
    samples_or_values : object
        samples_or_values value.
    comp : object
        comp value.
    sample_value : object
        sample_value value.
    """
    params = {
        param_name: sample_value(samples_or_values, custom_component_param_site(comp, param_name), default=0.0)
        for param_name in comp.parameter_priors
    }
    return jnp.asarray(comp.evaluate(wave, params, comp.metadata), dtype=jnp.float64)


def _evaluate_custom_line_component_jax(wave, samples_or_values, comp, sample_value):
    """Evaluate one custom line component from a sample/value mapping.

    Parameters
    ----------
    wave : object
        wave value.
    samples_or_values : object
        samples_or_values value.
    comp : object
        comp value.
    sample_value : object
        sample_value value.
    """
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
    host_basis_jax: Any | None = None
    t_obs_gyr: float | None = None


def _map_logzsol_to_dsps_lgmet(logzsol_grid: Sequence[float], ssp_lgmet: np.ndarray) -> np.ndarray:
    """Map fitting metallicity grid to DSPS metallicity convention.

    Parameters
    ----------
    logzsol_grid : object
        logzsol_grid value.
    ssp_lgmet : object
        ssp_lgmet value.
    """
    logzsol_grid = np.asarray(logzsol_grid, dtype=float)
    ssp_lgmet = np.asarray(ssp_lgmet, dtype=float)

    # DSPS metallicity grids are often log10(Z), while fitting grids are usually log10(Z/Zsun).
    # Select the transform that best matches the available DSPS metallicity grid.
    cand_direct = logzsol_grid
    cand_shifted = logzsol_grid + np.log10(0.019)

    def mismatch(cand):
        """Return mean nearest-neighbor mismatch to DSPS metallicity grid.

        Parameters
        ----------
        cand : object
            cand value.
        """
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
    z_qso: float = 0.0,
    build_physical_host_basis: bool = True,
    template_norms: Sequence[float] | None = None,
) -> FSPSTemplateGrid:
    """Build a host-galaxy SSP template matrix on the observed wavelength grid.

    Parameters
    ----------
    wave_out : object
        wave_out value.
    age_grid_gyr : object
        age_grid_gyr value.
    logzsol_grid : object
        logzsol_grid value.
    imf_type : object
        imf_type value.
    zcontinuous : object
        zcontinuous value.
    sfh : object
        sfh value.
    dsps_ssp_fn : object
        dsps_ssp_fn value.
    z_qso : object
        z_qso value.
    build_physical_host_basis : object
        build_physical_host_basis value.
    template_norms : object
        template_norms value.
    """
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
    template_norms_arr = None if template_norms is None else np.asarray(template_norms, dtype=float)
    expected_templates = int(age_grid_gyr.size * logzsol_grid.size)
    if template_norms_arr is not None and template_norms_arr.size != expected_templates:
        raise ValueError(
            "template_norms must match the age x metallicity template grid: "
            f"got {template_norms_arr.size}, expected {expected_templates}."
        )
    target_lg_age = np.log10(np.clip(age_grid_gyr, 1e-5, None))
    target_lgmet = _map_logzsol_to_dsps_lgmet(logzsol_grid, ssp_lgmet)

    tmpl = []
    meta = []
    itemp = 0
    for i_z, logz in enumerate(logzsol_grid):
        imet = int(np.argmin(np.abs(ssp_lgmet - target_lgmet[i_z])))
        for i_a, age in enumerate(age_grid_gyr):
            iage = int(np.argmin(np.abs(ssp_lg_age_gyr - target_lg_age[i_a])))
            spec_native = np.asarray(ssp_flux[imet, iage, :], dtype=float)
            spec_interp = np.interp(wave_out, ssp_wave, spec_native, left=0.0, right=0.0)
            if template_norms_arr is None:
                norm = np.nanmedian(np.abs(spec_interp))
                if not np.isfinite(norm) or norm <= 0:
                    norm = 1.0
            else:
                norm = float(template_norms_arr[itemp])
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("template_norms entries must be finite and positive.")
            spec_interp = spec_interp / norm
            tmpl.append(spec_interp)
            meta.append({
                'tage_gyr': float(age),
                'logzsol': float(logz),
                'norm': float(norm),
                'dsps_lgmet': float(ssp_lgmet[imet]),
                'dsps_lg_age_gyr': float(ssp_lg_age_gyr[iage]),
            })
            itemp += 1

    templates = np.column_stack(tmpl)
    t_obs_gyr = _cosmic_age_gyr(z_qso)
    host_basis_jax = (
        build_host_basis_jax(
            wave_out,
            dsps_ssp_fn=dsps_ssp_fn,
            t_obs_gyr=t_obs_gyr,
        )
        if build_physical_host_basis
        else None
    )

    return FSPSTemplateGrid(
        wave=wave_out,
        templates=templates,
        template_meta=meta,
        age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
        logzsol_grid=np.asarray(logzsol_grid, dtype=float),
        host_basis_jax=host_basis_jax,
        t_obs_gyr=t_obs_gyr,
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
    template_norms: Sequence[float] | None = None,
    n_draws: int | None = None,
    return_components: bool = True,
    decompose_host: bool = True,
) -> Dict[str, Any]:
    """Rebuild posterior continuum components on an arbitrary rest-frame grid.

    Parameters
    ----------
    wave_out : object
        wave_out value.
    samples : object
        samples value.
    pred_out : object
        pred_out value.
    age_grid_gyr : object
        age_grid_gyr value.
    logzsol_grid : object
        logzsol_grid value.
    dsps_ssp_fn : object
        dsps_ssp_fn value.
    prior_config : object
        prior_config value.
    fit_poly : object
        fit_poly value.
    fit_poly_order : object
        fit_poly_order value.
    fit_reddening : object
        fit_reddening value.
    fe_uv_wave : object
        fe_uv_wave value.
    fe_uv_flux : object
        fe_uv_flux value.
    fe_op_wave : object
        fe_op_wave value.
    fe_op_flux : object
        fe_op_flux value.
    custom_components : object
        custom_components value.
    template_norms : object
        template_norms value.
    n_draws : object
        n_draws value.
    return_components : object
        return_components value.
    decompose_host : object
        decompose_host value.
    """
    wave_out = np.asarray(wave_out, dtype=float)
    if wave_out.ndim != 1 or wave_out.size < 2 or not np.all(np.isfinite(wave_out)):
        raise ValueError("wave_out must be a finite 1D wavelength grid.")

    if decompose_host:
        fsps_grid = build_fsps_template_grid(
            wave_out=wave_out,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
            template_norms=template_norms,
        )
        templates = np.asarray(fsps_grid.templates, dtype=float)
    else:
        n_templates = int(len(tuple(age_grid_gyr)) * len(tuple(logzsol_grid)))
        templates = np.zeros((wave_out.size, n_templates), dtype=float)
    lnwave = np.log(wave_out)
    custom_components = normalize_custom_components(custom_components)
    convolution_method = str(prior_config.get("convolution_method", "fft")).lower()

    n_total = int(np.asarray(next(iter(samples.values()))).shape[0]) if len(samples) > 0 else 0
    if n_total == 0:
        raise RuntimeError("Posterior samples are empty.")
    n_use = n_total if n_draws is None else max(1, min(int(n_draws), n_total))
    sl = slice(0, n_use)

    pl_norm = np.asarray(samples.get('PL_norm', np.zeros(n_total)), dtype=float)[sl]
    pl_slope = np.asarray(samples.get('PL_slope', np.zeros(n_total)), dtype=float)[sl]
    gal_v = np.asarray(samples.get('gal_v_kms', np.zeros(n_total)), dtype=float)[sl]
    gal_sigma = np.asarray(samples.get('gal_sigma_kms', np.full(n_total, 150.0)), dtype=float)[sl]

    if pred_out is not None and 'fsps_weights' in pred_out:
        fsps_weights = np.asarray(pred_out['fsps_weights'], dtype=float)[sl]
    else:
        fsps_weights = np.zeros((n_use, templates.shape[1]), dtype=float)
    if fsps_weights.ndim == 1:
        fsps_weights = fsps_weights[:, np.newaxis]
    if fsps_weights.ndim != 2 or fsps_weights.shape[1] != templates.shape[1]:
        raise RuntimeError(
            "Posterior fsps_weights shape is incompatible with the reconstruction "
            f"template grid: got weights shape {fsps_weights.shape}, expected "
            f"second dimension {templates.shape[1]} for decompose_host={bool(decompose_host)}."
        )

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

    if prior_config.get("PL_pivot", None) is None and np.any(np.asarray(pl_norm, dtype=float) != 0.0):
        raise ValueError(
            "Posterior reconstruction with power-law samples requires "
            "prior_config['PL_pivot'] from the fitted wavelength grid."
        )
    pl_pivot = float(np.asarray(_resolve_pl_pivot(wave_out, prior_config), dtype=float))
    reddening_a2500 = np.asarray(samples.get('reddening_a2500', np.zeros(n_total)), dtype=float)[sl]
    reddening_uv_ref = float(prior_config.get('reddening_uv_ref', 2500.0))
    reddening_alpha = float(prior_config.get('reddening_alpha', 1.2))
    if fit_poly and fit_poly_order > 0:
        poly_coeffs = np.column_stack([
            np.asarray(samples.get(f'poly_c{k}', np.zeros(n_total)), dtype=float)[sl]
            for k in range(1, fit_poly_order + 1)
        ])
    else:
        poly_coeffs = np.zeros((n_use, 0), dtype=float)

    wave_j = jnp.asarray(wave_out, dtype=jnp.float64)
    lnwave_j = jnp.asarray(lnwave, dtype=jnp.float64)
    templates_j = jnp.asarray(templates, dtype=jnp.float64)
    fsps_weights_j = jnp.asarray(fsps_weights, dtype=jnp.float64)
    poly_coeffs_j = jnp.asarray(poly_coeffs, dtype=jnp.float64)
    poly_powers_j = None
    if fit_poly and fit_poly_order > 0:
        w0 = float(np.asarray(_resolve_poly_pivot(wave_out, prior_config, require_configured=True), dtype=float))
        x = (wave_out - w0) / max(w0, 1.0)
        poly_powers_j = jnp.asarray(
            np.vstack([x ** k for k in range(1, fit_poly_order + 1)]),
            dtype=jnp.float64,
        )

    def _one_builtin_components(
        weights_i,
        pl_norm_i,
        pl_slope_i,
        gal_v_i,
        gal_sigma_i,
        fe_uv_norm_i,
        fe_op_norm_i,
        fe_uv_fwhm_i,
        fe_op_fwhm_i,
        fe_uv_shift_i,
        fe_op_shift_i,
        balmer_norm_i,
        balmer_tau_i,
        balmer_vel_i,
        reddening_a2500_i,
        poly_coeffs_i,
    ):
        """Evaluate built-in host, continuum, Fe II, Balmer, and polynomial terms for one draw.

        Parameters
        ----------
        weights_i : object
            weights_i value.
        pl_norm_i : object
            pl_norm_i value.
        pl_slope_i : object
            pl_slope_i value.
        gal_v_i : object
            gal_v_i value.
        gal_sigma_i : object
            gal_sigma_i value.
        fe_uv_norm_i : object
            fe_uv_norm_i value.
        fe_op_norm_i : object
            fe_op_norm_i value.
        fe_uv_fwhm_i : object
            fe_uv_fwhm_i value.
        fe_op_fwhm_i : object
            fe_op_fwhm_i value.
        fe_uv_shift_i : object
            fe_uv_shift_i value.
        fe_op_shift_i : object
            fe_op_shift_i value.
        balmer_norm_i : object
            balmer_norm_i value.
        balmer_tau_i : object
            balmer_tau_i value.
        balmer_vel_i : object
            balmer_vel_i value.
        reddening_a2500_i : object
            reddening_a2500_i value.
        poly_coeffs_i : object
            poly_coeffs_i value.
        """
        host_intrinsic = templates_j @ weights_i
        host_model = _shift_and_broaden_single_spectrum_lnlam(
            lnwave_j,
            host_intrinsic,
            gal_v_i,
            gal_sigma_i,
            convolution_method=convolution_method,
        )

        pl_model = _powerlaw_jax(
            wave_j,
            pl_norm=pl_norm_i,
            pl_slope=pl_slope_i,
            pivot=pl_pivot,
        )
        reddening_atten = jnp.ones_like(wave_j)
        if fit_reddening:
            reddening_atten = _smc_like_reddening_jax(
                wave_j,
                reddening_a2500_i,
                uv_ref=reddening_uv_ref,
                alpha=reddening_alpha,
            )
            pl_model = pl_model * reddening_atten
        fe_uv_model = _fe_template_component(
            wave_j,
            jnp.asarray(fe_uv_wave, dtype=jnp.float64),
            jnp.asarray(fe_uv_flux, dtype=jnp.float64),
            fe_uv_norm_i,
            fe_uv_fwhm_i,
            fe_uv_shift_i,
            convolution_method=convolution_method,
        )
        fe_op_model = _fe_template_component(
            wave_j,
            jnp.asarray(fe_op_wave, dtype=jnp.float64),
            jnp.asarray(fe_op_flux, dtype=jnp.float64),
            fe_op_norm_i,
            fe_op_fwhm_i,
            fe_op_shift_i,
            convolution_method=convolution_method,
        )
        bc_model = _balmer_continuum_jax(
            wave_j,
            balmer_norm_i,
            15000.0,
            balmer_tau_i,
            balmer_vel_i,
            convolution_method=convolution_method,
        )
        if fit_reddening:
            fe_uv_model = fe_uv_model * reddening_atten
            fe_op_model = fe_op_model * reddening_atten
            bc_model = bc_model * reddening_atten

        poly_model = jnp.ones_like(wave_j)
        if fit_poly:
            poly_base = jnp.ones_like(wave_j)
            if fit_poly_order > 0:
                poly_base = poly_base + jnp.sum(poly_coeffs_i[:, None] * poly_powers_j, axis=0)
            poly_model = jnp.clip(poly_base, 0.2, 5.0)

        host_model = host_model * poly_model
        pl_model = pl_model * poly_model
        fe_uv_model = fe_uv_model * poly_model
        fe_op_model = fe_op_model * poly_model
        bc_model = bc_model * poly_model
        continuum_model = pl_model + fe_uv_model + fe_op_model + bc_model + host_model
        return host_model, pl_model, fe_uv_model, fe_op_model, bc_model, continuum_model, poly_model, reddening_atten

    (
        host_draws,
        pl_draws,
        fe_uv_draws,
        fe_op_draws,
        bc_draws,
        continuum_draws,
        poly_draws,
        reddening_atten_draws,
    ) = jax.vmap(_one_builtin_components)(
        fsps_weights_j,
        jnp.asarray(pl_norm, dtype=jnp.float64),
        jnp.asarray(pl_slope, dtype=jnp.float64),
        jnp.asarray(gal_v, dtype=jnp.float64),
        jnp.asarray(gal_sigma, dtype=jnp.float64),
        jnp.asarray(fe_uv_norm, dtype=jnp.float64),
        jnp.asarray(fe_op_norm, dtype=jnp.float64),
        jnp.asarray(fe_uv_fwhm, dtype=jnp.float64),
        jnp.asarray(fe_op_fwhm, dtype=jnp.float64),
        jnp.asarray(fe_uv_shift, dtype=jnp.float64),
        jnp.asarray(fe_op_shift, dtype=jnp.float64),
        jnp.asarray(balmer_norm, dtype=jnp.float64),
        jnp.asarray(balmer_tau, dtype=jnp.float64),
        jnp.asarray(balmer_vel, dtype=jnp.float64),
        jnp.asarray(reddening_a2500, dtype=jnp.float64),
        poly_coeffs_j,
    )

    component_draws = {
        'host': np.asarray(host_draws, dtype=float),
        'PL': np.asarray(pl_draws, dtype=float),
        'Fe_uv': np.asarray(fe_uv_draws, dtype=float),
        'Fe_op': np.asarray(fe_op_draws, dtype=float),
        'Balmer_cont': np.asarray(bc_draws, dtype=float),
        'continuum': np.asarray(continuum_draws, dtype=float),
    }
    poly_draws_np = np.asarray(poly_draws, dtype=float)
    reddening_atten_draws_np = np.asarray(reddening_atten_draws, dtype=float)
    custom_total_draws = np.zeros((n_use, wave_out.size), dtype=float)
    for comp in custom_components:
        comp_draws = np.zeros((n_use, wave_out.size), dtype=float)
        for i in range(n_use):
            def _sample_value(samples_dict, key, default=0.0):
                """Read one custom-component parameter draw with a fallback value.

                Parameters
                ----------
                samples_dict : object
                    samples_dict value.
                key : object
                    key value.
                default : object
                    default value.
                """
                val = float(np.asarray(samples_dict.get(key, np.full(n_total, default)), dtype=float)[sl][i])
                return val

            comp_draw = np.asarray(
                _evaluate_custom_component_jax(wave_out, samples, comp, _sample_value),
                dtype=float,
            ) * reddening_atten_draws_np[i] * poly_draws_np[i]
            comp_draws[i] = comp_draw
            custom_total_draws[i] = custom_total_draws[i] + comp_draw
        component_draws[comp.output_name] = comp_draws

    if custom_components:
        component_draws['continuum'] = component_draws['continuum'] + custom_total_draws

    output_draws = component_draws if return_components else {'continuum': component_draws['continuum']}
    return {
        'wave': wave_out,
        'draws': output_draws,
        'median': {key: np.median(val, axis=0) for key, val in output_draws.items()},
    }


def _extract_line_table_from_prior_config(prior_config: Dict[str, Any] | None):
    """Extract line-table priors from the canonical ``line.table`` layout.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    """
    prior_config = _materialize_prior_mapping(prior_config)
    line_cfg = prior_config.get('line', None)
    if isinstance(line_cfg, dict):
        if 'table' in line_cfg:
            return line_cfg['table']
    return None


def _compress_group_ids(ids: np.ndarray, labels: Sequence[str] | None = None) -> Tuple[np.ndarray, Dict[Any, int]]:
    """Compress sparse positive tie ids into contiguous group indices.

    Parameters
    ----------
    ids : object
        ids value.
    labels : object
        labels value.
    """
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
    """Build tied-line metadata arrays used by the NumPyro line model.

    Parameters
    ----------
    linelist : object
        linelist value.
    wave : object
        wave value.
    """
    def _to_records(obj):
        """Normalize line table inputs to `list[dict]` records.

        Parameters
        ----------
        obj : object
            obj value.
        """
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
    ln_wmin = np.log(max(wmin, 1e-300))
    ln_wmax = np.log(max(wmax, 1e-300))
    support_nsigma = 5.0
    for row in records:
        lam = float(row['lambda'])
        ln0 = np.log(max(lam, 1e-300))
        voff = abs(float(row.get('voff', 0.0)))
        sig_max = max(float(row.get('maxsig', row.get('inisig', 1e-5))), 1e-5)
        ln_support = voff + support_nsigma * sig_max
        if (ln0 + ln_support) >= ln_wmin and (ln0 - ln_support) <= ln_wmax:
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
        ngauss = int(row.get('ngauss', 1))
        linename = str(row.get('linename', f"line_{row['lambda']:.1f}"))
        base_compname = str(row.get('compname', linename))
        for i in range(ngauss):
            ln0 = np.log(float(row['lambda']))
            voff = float(row['voff'])
            dln = voff
            init_amp = float(row['inisca'])
            if not np.isfinite(init_amp) or init_amp <= 0.0:
                init_amp = abs(float(row.get('fvalue', 0.0)))
            ln_lambda0.append(ln0)
            line_lambda.append(float(row['lambda']))
            amp_init.append(init_amp)
            amp_min.append(float(row['minsca']))
            amp_max.append(float(row['maxsca']))
            sig_init.append(max(float(row['inisig']), 1e-5))
            sig_min.append(max(float(row['minsig']), 1e-5))
            sig_max.append(max(float(row['maxsig']), 1e-5))
            dmu_min.append(-dln)
            dmu_max.append(+dln)
            names.append(f"{linename}_{i+1}")
            vindex.append(int(row['vindex']))
            windex.append(int(row['windex']))
            findex.append(int(row['findex']))
            fvalue.append(float(row['fvalue']))
            if ngauss > 1:
                compnames.append(f"{base_compname}:{linename}:{i + 1}")
            else:
                compnames.append(base_compname)

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
        if amp_max_group[gid] <= amp_min_group[gid]:
            amp_max_group[gid] = max(amp_min_group[gid] * 1.1, amp_min_group[gid] + 1.0e-4)
        amp_eps = max(1.0e-8 * (amp_max_group[gid] - amp_min_group[gid]), AMPLITUDE_FLOOR)
        amp_init_group[gid] = np.clip(
            amp_init_group[gid],
            amp_min_group[gid] + amp_eps,
            amp_max_group[gid] - amp_eps,
        )

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
                         fe_uv_flux_on_wave=None, fe_op_flux_on_wave=None,
                         balmer_bb_shape=None, balmer_tau_shape=None, balmer_below_edge=None,
                         return_line_components=True,
                         emit_deterministics=True,
                         custom_components: Sequence[CustomComponentSpec] | None = None,
                         custom_line_components: Sequence[CustomLineComponentSpec] | None = None):
    """Joint AGN+host spectral forward model for NumPyro inference.

    Parameters
    ----------
    wave : object
        wave value.
    flux : object
        flux value.
    err : object
        err value.
    conti_priors : object
        conti_priors value.
    tied_line_meta : object
        tied_line_meta value.
    fsps_grid : object
        fsps_grid value.
    fe_uv_wave : object
        fe_uv_wave value.
    fe_uv_flux : object
        fe_uv_flux value.
    fe_op_wave : object
        fe_op_wave value.
    fe_op_flux : object
        fe_op_flux value.
    use_lines : object
        use_lines value.
    prior_config : object
        prior_config value.
    decompose_host : object
        decompose_host value.
    fit_pl : object
        fit_pl value.
    fit_fe : object
        fit_fe value.
    fit_bc : object
        fit_bc value.
    fit_poly : object
        fit_poly value.
    fit_poly_order : object
        fit_poly_order value.
    fit_reddening : object
        fit_reddening value.
    z_qso : object
        z_qso value.
    psf_mags : object
        psf_mags value.
    psf_mag_errs : object
        psf_mag_errs value.
    psf_filter_curves : object
        psf_filter_curves value.
    use_psf_phot : object
        use_psf_phot value.
    fe_uv_flux_on_wave : object
        fe_uv_flux_on_wave value.
    fe_op_flux_on_wave : object
        fe_op_flux_on_wave value.
    balmer_bb_shape : object
        balmer_bb_shape value.
    balmer_tau_shape : object
        balmer_tau_shape value.
    balmer_below_edge : object
        balmer_below_edge value.
    return_line_components : object
        return_line_components value.
    emit_deterministics : object
        emit_deterministics value.
    custom_components : object
        custom_components value.
    custom_line_components : object
        custom_line_components value.
    """
    has_observed_flux = flux is not None
    wave = _np_to_jnp(wave)
    flux = _np_to_jnp(flux)
    err = _np_to_jnp(err)
    lnwave = jnp.log(wave)
    templates = _np_to_jnp(fsps_grid.templates)
    fe_uv_wave = _np_to_jnp(fe_uv_wave)
    fe_uv_flux = _np_to_jnp(fe_uv_flux)
    fe_op_wave = _np_to_jnp(fe_op_wave)
    fe_op_flux = _np_to_jnp(fe_op_flux)
    fe_uv_flux_on_wave = None if fe_uv_flux_on_wave is None else _np_to_jnp(fe_uv_flux_on_wave)
    fe_op_flux_on_wave = None if fe_op_flux_on_wave is None else _np_to_jnp(fe_op_flux_on_wave)
    balmer_static_terms = None
    if balmer_bb_shape is not None and balmer_tau_shape is not None and balmer_below_edge is not None:
        balmer_static_terms = (
            _np_to_jnp(balmer_bb_shape),
            _np_to_jnp(balmer_tau_shape),
            jnp.asarray(balmer_below_edge, dtype=bool),
        )
    z_qso = jnp.asarray(z_qso, dtype=jnp.float64)
    prior_config = _materialize_prior_mapping(prior_config)
    convolution_method = str(prior_config.get("convolution_method", "fft")).lower()
    custom_components = normalize_custom_components(custom_components)
    custom_line_components = normalize_custom_line_components(custom_line_components)
    bal_absorption_components = tuple(
        comp for comp in custom_components if _is_multiplicative_bal_component(comp)
    )
    additive_custom_components = tuple(
        comp for comp in custom_components if not _is_multiplicative_bal_component(comp)
    )
    use_psf_phot = (
        bool(use_psf_phot)
        and psf_mags is not None
        and psf_mag_errs is not None
        and psf_filter_curves is not None
    )
    return_line_components = bool(return_line_components)
    emit_deterministics = bool(emit_deterministics)

    host_sfh_model = str(prior_config.get("host_sfh_model", "flexible")).lower()
    physical_delayed_host = (
        decompose_host
        and host_sfh_model in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}
        and getattr(fsps_grid, "host_basis_jax", None) is not None
    )

    if decompose_host and not physical_delayed_host:
        cont_norm = _sample_positive_distribution(
            prior_config,
            value_key='cont_norm',
            log_key='log_cont_norm',
            default_value_distribution=dist.LogNormal(0.0, 1.0),
            default_log_distribution=dist.Normal(0.0, 1.0),
        )
        if isinstance(prior_config.get('log_frac_host', None), dict) and ('df' in prior_config['log_frac_host']):
            log_frac_host_df = float(prior_config['log_frac_host']['df'])
        else:
            log_frac_host_df = float(prior_config.get('log_frac_host_df', 3.0))
        log_frac_host_loc, log_frac_host_scale = _prior_loc_scale(prior_config, 'log_frac_host')
        host_redshift_prior_weight, host_redshift_prior_loc_offset, host_redshift_prior_scale_mult, host_redshift_prior_df_eff = _host_redshift_prior_params(prior_config, z_qso)
        log_frac_host_loc_eff = log_frac_host_loc + host_redshift_prior_loc_offset
        log_frac_host_scale_eff = jnp.maximum(log_frac_host_scale * host_redshift_prior_scale_mult, 1e-6)
        log_frac_host_df_eff = jnp.asarray(log_frac_host_df) if host_redshift_prior_df_eff is None else jnp.maximum(host_redshift_prior_df_eff, 1e-6)
        host_redshift_prior_enabled = bool(prior_config.get("host_redshift_prior", {}).get("enabled", True))
        if (
            not host_redshift_prior_enabled
            and host_redshift_prior_df_eff is None
            and _prior_family(prior_config, 'log_frac_host', 'student_t') not in {"normal", "gaussian"}
        ):
            log_frac_host_sample = _sample_prior(
                prior_config,
                'log_frac_host',
                dist.StudentT(df=log_frac_host_df_eff, loc=log_frac_host_loc_eff, scale=log_frac_host_scale_eff),
            )
        else:
            log_frac_host_sample = numpyro.sample(
                'log_frac_host',
                dist.StudentT(df=log_frac_host_df_eff, loc=log_frac_host_loc_eff, scale=log_frac_host_scale_eff),
            )
        frac_host_sample = jax.nn.sigmoid(log_frac_host_sample)
        if emit_deterministics:
            numpyro.deterministic('host_redshift_prior_weight', host_redshift_prior_weight)
            numpyro.deterministic('host_redshift_prior_loc_eff', log_frac_host_loc_eff)
            numpyro.deterministic('host_redshift_prior_scale_eff', log_frac_host_scale_eff)
            numpyro.deterministic('host_redshift_prior_df_eff', log_frac_host_df_eff)
        host_amp = cont_norm * frac_host_sample
    else:
        host_amp = jnp.asarray(jnp.nan)
        log_frac_host_sample = jnp.asarray(jnp.nan)
        frac_host_sample = jnp.asarray(jnp.nan)
    pl_pivot = _resolve_pl_pivot(wave, prior_config)
    if fit_pl:
        pl_norm = _sample_prior(prior_config, 'PL_norm', dist.HalfNormal(1.0))
        pl_slope = _sample_prior(prior_config, 'PL_slope', dist.Normal(0.0, 1.0))
        reddening_a2500 = (
            _sample_positive_distribution(
                prior_config,
                value_key='reddening_a2500',
                log_key='log_reddening_a2500',
                default_value_distribution=dist.LogNormal(np.log(0.1), 1.0),
                default_log_distribution=dist.Normal(np.log(0.1), 1.0),
                default_to_log=True,
            )
            if fit_reddening else jnp.asarray(0.0)
        )
    else:
        pl_norm = jnp.asarray(0.0)
        pl_slope = jnp.asarray(0.0)
        reddening_a2500 = jnp.asarray(0.0)

    if fit_fe:
        fe_uv_norm = _sample_positive_distribution(
            prior_config,
            value_key='Fe_uv_norm',
            log_key='log_Fe_uv_norm',
            default_value_distribution=dist.LogNormal(np.log(1.0e-3), 2.0),
            default_log_distribution=dist.Normal(np.log(1.0e-3), 2.0),
        )
        log_fe_op_over_uv = _sample_prior(prior_config, 'log_Fe_op_over_uv', dist.Normal(0.0, 1.0))
        fe_op_norm = fe_uv_norm * jnp.exp(log_fe_op_over_uv)
        fe_uv_fwhm = _sample_positive_distribution(
            prior_config,
            value_key='Fe_uv_FWHM',
            log_key='log_Fe_uv_FWHM',
            default_value_distribution=dist.LogNormal(np.log(3000.0), 0.5),
            default_log_distribution=dist.Normal(np.log(3000.0), 0.5),
        )
        fe_op_fwhm = _sample_positive_distribution(
            prior_config,
            value_key='Fe_op_FWHM',
            log_key='log_Fe_op_FWHM',
            default_value_distribution=dist.LogNormal(np.log(3000.0), 0.5),
            default_log_distribution=dist.Normal(np.log(3000.0), 0.5),
        )
        fe_uv_shift = _sample_prior(prior_config, 'Fe_uv_shift', dist.Normal(0.0, 0.01))
        fe_op_shift = _sample_prior(prior_config, 'Fe_op_shift', dist.Normal(0.0, 0.01))
    else:
        fe_uv_norm = jnp.asarray(0.0)
        fe_op_norm = jnp.asarray(0.0)
        fe_uv_fwhm = jnp.asarray(3000.0)
        fe_op_fwhm = jnp.asarray(3000.0)
        fe_uv_shift = jnp.asarray(0.0)
        fe_op_shift = jnp.asarray(0.0)

    if fit_bc:
        balmer_norm = _sample_positive_distribution(
            prior_config,
            value_key='Balmer_norm',
            log_key='log_Balmer_norm',
            default_value_distribution=dist.LogNormal(np.log(1.0e-3), 2.0),
            default_log_distribution=dist.Normal(np.log(1.0e-3), 2.0),
        )
        balmer_te = jnp.asarray(15000.0)
        balmer_tau = _sample_positive_distribution(
            prior_config,
            value_key='Balmer_Tau',
            log_key='log_Balmer_Tau',
            default_value_distribution=dist.LogNormal(np.log(0.5), 0.25),
            default_log_distribution=dist.Normal(np.log(0.5), 0.25),
        )
        balmer_vel = _sample_positive_distribution(
            prior_config,
            value_key='Balmer_vel',
            log_key='log_Balmer_vel',
            default_value_distribution=dist.LogNormal(np.log(3000.0), 0.25),
            default_log_distribution=dist.Normal(np.log(3000.0), 0.25),
        )
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
            reddening_a2500,
            uv_ref=float(prior_config.get('reddening_uv_ref', 2500.0)),
            alpha=float(prior_config.get('reddening_alpha', 1.2)),
        )
        if fit_reddening else jnp.ones_like(wave)
    )
    if fit_fe:
        fe_uv_model_intrinsic = _fe_template_component(
            wave,
            fe_uv_wave,
            fe_uv_flux,
            fe_uv_norm,
            fe_uv_fwhm,
            fe_uv_shift,
            template_on_wave=fe_uv_flux_on_wave,
            convolution_method=convolution_method,
        )
        fe_op_model_intrinsic = _fe_template_component(
            wave,
            fe_op_wave,
            fe_op_flux,
            fe_op_norm,
            fe_op_fwhm,
            fe_op_shift,
            template_on_wave=fe_op_flux_on_wave,
            convolution_method=convolution_method,
        )
    else:
        fe_uv_model_intrinsic = jnp.zeros_like(wave)
        fe_op_model_intrinsic = jnp.zeros_like(wave)
    if fit_bc:
        bc_model_intrinsic = _balmer_continuum_jax(
            wave,
            balmer_norm,
            balmer_te,
            balmer_tau,
            balmer_vel,
            balmer_static_terms=balmer_static_terms,
            convolution_method=convolution_method,
        )
    else:
        bc_model_intrinsic = jnp.zeros_like(wave)
    pl_model = pl_model_intrinsic * reddening_atten
    fe_uv_model = fe_uv_model_intrinsic * reddening_atten
    fe_op_model = fe_op_model_intrinsic * reddening_atten
    bc_model = bc_model_intrinsic * reddening_atten
    custom_models = {}
    custom_total_model = jnp.zeros_like(wave)
    for comp in additive_custom_components:
        def _sample_value(sample_dict, key, default=0.0):
            """Sample one custom continuum-component parameter from prior config.

            Parameters
            ----------
            sample_dict : object
                sample_dict value.
            key : object
                key value.
            default : object
                default value.
            """
            cfg = prior_config.get(key, None)
            if cfg is None:
                return default
            return _sample_from_prior_config(key, cfg)

        custom_model_intrinsic = _evaluate_custom_component_jax(wave, prior_config, comp, _sample_value)
        custom_model = custom_model_intrinsic * reddening_atten
        custom_models[comp.output_name] = custom_model
        custom_total_model = custom_total_model + custom_model
    poly_model = jnp.ones_like(wave)
    if fit_poly:
        poly_order = int(max(fit_poly_order, 0))
        w0 = _resolve_poly_pivot(wave, prior_config)
        x = (wave - w0) / jnp.maximum(w0, 1.0)
        # Global low-order tilt
        poly_base = jnp.ones_like(wave)
        for k in range(1, poly_order + 1):
            ck = _sample_prior(prior_config, f'poly_c{k}', dist.Normal(0.0, 0.1))
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
                    reddening_a2500,
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
    host_aperture_scale = jnp.asarray(1.0, dtype=jnp.float64)
    if decompose_host:
        log_host_aperture_scale = _sample_log_host_aperture_scale(prior_config)
        host_aperture_scale = jnp.exp(log_host_aperture_scale)
        if host_sfh_model in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}:
            gal_intrinsic_total, fsps_weights, fsps_weights_frac = _delayed_sfh_host_spectrum(
                fsps_grid,
                prior_config,
                host_amp,
                z_qso,
            )
        elif host_sfh_model in {"flexible", "free", "template_weights", "ssp_weights"}:
            tau_host = _sample_prior(prior_config, 'tau_host', dist.HalfNormal(1.0))
            tau_host_eff = jnp.maximum(tau_host, 1e-6)
            raw_w_loc = _flexible_host_raw_weight_locs(fsps_grid, prior_config, ntemp)
            raw_w = numpyro.sample('fsps_weights_raw', dist.Normal(raw_w_loc, tau_host_eff))
            fsps_weights_frac = jax.nn.softmax(raw_w)
            fsps_weights_total = host_amp * fsps_weights_frac
            fsps_weights = host_aperture_scale * fsps_weights_total
            gal_intrinsic_total = jnp.dot(templates, fsps_weights_total)
        else:
            raise ValueError("host_sfh_model must be one of: 'flexible', 'delayed'.")
        gal_v_kms = _sample_prior(prior_config, 'gal_v_kms', dist.Normal(0.0, 150.0))
        gal_sigma_kms = _sample_positive_distribution(
            prior_config,
            value_key='gal_sigma_kms',
            log_key='log_gal_sigma_kms',
            default_value_distribution=dist.LogNormal(np.log(150.0), 0.4),
            default_log_distribution=dist.Normal(np.log(150.0), 0.4),
            default_to_log=True,
        )
        gal_model_intrinsic_total = _shift_and_broaden_single_spectrum_lnlam(
            lnwave,
            gal_intrinsic_total,
            gal_v_kms,
            gal_sigma_kms,
            convolution_method=convolution_method,
        )
        gal_model_intrinsic = host_aperture_scale * gal_model_intrinsic_total
    else:
        fsps_weights_frac = jnp.zeros((ntemp,))
        fsps_weights = jnp.zeros((ntemp,))
        gal_model_intrinsic_total = jnp.zeros_like(wave)
        gal_model_intrinsic = jnp.zeros_like(wave)

    custom_line_models = {}
    custom_line_broad_intrinsic = jnp.zeros_like(wave)
    custom_line_narrow_intrinsic = jnp.zeros_like(wave)
    line_component_profiles = jnp.zeros((0, wave.shape[0]), dtype=wave.dtype)
    line_component_broad_mask = jnp.zeros((0,), dtype=wave.dtype)
    for comp in custom_line_components:
        def _sample_line_value(sample_dict, key, default=0.0):
            """Sample one custom line-component parameter from prior config.

            Parameters
            ----------
            sample_dict : object
                sample_dict value.
            key : object
                key value.
            default : object
                default value.
            """
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

    line_components_are_split = return_line_components or use_psf_phot or fit_reddening
    if use_lines and tied_line_meta['n_lines'] > 0:
        dmu_group, sig_group, amp_group = _sample_tied_line_groups(
            tied_line_meta,
            prior_config,
        )

        vgroup = _line_meta_array(tied_line_meta, 'vgroup', jax_key='vgroup_jax', dtype=jnp.int32)
        wgroup = _line_meta_array(tied_line_meta, 'wgroup', jax_key='wgroup_jax', dtype=jnp.int32)
        fgroup = _line_meta_array(tied_line_meta, 'fgroup', jax_key='fgroup_jax', dtype=jnp.int32)
        dmu = dmu_group[vgroup]
        sigs = sig_group[wgroup]
        amps = amp_group[fgroup] * _line_meta_array(tied_line_meta, 'flux_ratio', jax_key='flux_ratio_jax')
        mus = tied_line_meta['ln_lambda0'] + dmu
        line_component_broad_mask = _line_meta_array(tied_line_meta, 'broad_mask', jax_key='broad_mask_jax')

        if line_components_are_split:
            (
                line_model_intrinsic,
                line_model_broad_intrinsic,
                line_model_narrow_intrinsic,
                line_component_profiles,
            ) = _split_many_gauss_lnlam(
                lnwave,
                amps,
                mus,
                sigs,
                line_component_broad_mask,
                return_profiles=bool(emit_deterministics),
            )
            if line_component_profiles is None:
                line_component_profiles = jnp.zeros((0, wave.shape[0]), dtype=wave.dtype)
            line_model_broad_intrinsic = line_model_broad_intrinsic + custom_line_broad_intrinsic
            line_model_narrow_intrinsic = line_model_narrow_intrinsic + custom_line_narrow_intrinsic
            line_model_intrinsic = line_model_broad_intrinsic + line_model_narrow_intrinsic
        else:
            line_model_intrinsic = _many_gauss_lnlam(lnwave, amps, mus, sigs) + custom_line_broad_intrinsic + custom_line_narrow_intrinsic
            line_model_broad_intrinsic = jnp.zeros_like(wave)
            line_model_narrow_intrinsic = jnp.zeros_like(wave)
            line_component_profiles = (
                amps[:, None] * jnp.exp(-0.5 * ((lnwave[None, :] - mus[:, None]) / sigs[:, None]) ** 2)
                if emit_deterministics
                else jnp.zeros((0, wave.shape[0]), dtype=wave.dtype)
            )
        if emit_deterministics:
            numpyro.deterministic('line_amp_per_component', amps)
            numpyro.deterministic('line_mu_per_component', mus)
            numpyro.deterministic('line_sig_per_component', sigs)
    else:
        line_model_broad_intrinsic = custom_line_broad_intrinsic
        line_model_narrow_intrinsic = custom_line_narrow_intrinsic
        line_model_intrinsic = custom_line_broad_intrinsic + custom_line_narrow_intrinsic

    gal_model_total = gal_model_intrinsic_total
    gal_model = gal_model_intrinsic
    line_model_broad = line_model_broad_intrinsic * reddening_atten
    line_model_narrow = line_model_narrow_intrinsic
    line_model = line_model_broad + line_model_narrow if line_components_are_split else line_model_intrinsic
    if line_component_profiles.shape[0] > 0:
        line_component_profiles = line_component_profiles * (
            line_component_broad_mask[:, None] * reddening_atten[None, :]
            + (1.0 - line_component_broad_mask[:, None])
        )
    custom_line_models = {
        comp.output_name: custom_line_models[comp.output_name]
        * (reddening_atten if comp.line_kind == 'broad' else 1.0)
        for comp in custom_line_components
    }
    if fit_poly:
        gal_model_total = gal_model_total * poly_model
        gal_model = gal_model * poly_model
        if line_components_are_split:
            line_model_broad = line_model_broad * poly_model
            line_model_narrow = line_model_narrow * poly_model
            line_model = line_model_broad + line_model_narrow
        else:
            line_model = line_model * poly_model
        line_component_profiles = line_component_profiles * poly_model[None, :]
        custom_line_models = {name: model * poly_model for name, model in custom_line_models.items()}

    if bal_absorption_components:
        bal_reference = agn_model + line_model_broad
        bal_transmission = jnp.ones_like(wave)
        bal_param_cache = {}
        for comp in bal_absorption_components:
            def _sample_bal_value(sample_dict, key, default=0.0):
                """Sample one BAL absorption parameter from prior config.

                Parameters
                ----------
                sample_dict : object
                    sample_dict value.
                key : object
                    key value.
                default : object
                    default value.
                """
                if key in bal_param_cache:
                    return bal_param_cache[key]
                cfg = prior_config.get(key, None)
                if cfg is None:
                    return default
                value = _sample_from_prior_config(key, cfg)
                bal_param_cache[key] = value
                return value

            bal_params = {
                param_name: _sample_bal_value(
                    prior_config,
                    custom_component_param_site(comp, param_name),
                    default=0.0,
                )
                for param_name in comp.parameter_priors
            }
            tau_profile = jnp.asarray(comp.evaluate(wave, bal_params, comp.metadata), dtype=jnp.float64)
            covering = _bal_covering_fraction(bal_params)
            component_transmission = 1.0 - covering * (1.0 - jnp.exp(-tau_profile))
            component_transmission = jnp.clip(component_transmission, 1.0e-6, 1.0)
            custom_models[comp.output_name] = bal_reference * (component_transmission - 1.0)
            bal_transmission = bal_transmission * component_transmission

        bal_transmission = jnp.clip(bal_transmission, 1.0e-6, 1.0)
        agn_model = agn_model * bal_transmission
        line_model_broad = line_model_broad * bal_transmission
        line_model = line_model_broad + line_model_narrow if line_components_are_split else line_model * bal_transmission
        if line_component_profiles.shape[0] > 0:
            line_component_profiles = line_component_profiles * (
                line_component_broad_mask[:, None] * bal_transmission[None, :]
                + (1.0 - line_component_broad_mask[:, None])
            )
        custom_line_models = {
            comp.output_name: custom_line_models[comp.output_name]
            * (bal_transmission if comp.line_kind == 'broad' else 1.0)
            for comp in custom_line_components
        }

    if decompose_host and not physical_delayed_host:
        frac_host = frac_host_sample
        log_frac_host = log_frac_host_sample
        host_amp_out = host_amp
    elif decompose_host:
        host_ref = jnp.abs(jnp.interp(pl_pivot, wave, gal_model, left=0.0, right=0.0))
        agn_ref = jnp.abs(jnp.interp(pl_pivot, wave, agn_model, left=0.0, right=0.0))
        frac_host = jnp.clip(host_ref / jnp.maximum(host_ref + agn_ref, 1.0e-30), 1.0e-12, 1.0 - 1.0e-12)
        log_frac_host = jnp.log(frac_host) - jnp.log1p(-frac_host)
        host_amp_out = jnp.where(jnp.isfinite(host_amp), host_amp, host_ref)
    else:
        frac_host = jnp.asarray(0.0)
        log_frac_host = jnp.asarray(-jnp.inf)
        host_amp_out = jnp.asarray(0.0)

    frac_jitter = _sample_prior(prior_config, 'frac_jitter', dist.HalfNormal(0.02))
    add_jitter = numpyro.sample('add_jitter', _halfnormal_prior(prior_config, 'add_jitter', 0.1, ref_scale=jnp.mean(err)))

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
    line_component_profiles_psf = line_component_profiles
    psf_model = agn_model_psf + gal_model_psf + line_model_psf
    if use_psf_phot:
        delta_m_psf = numpyro.sample('delta_m_psf_raw', dist.Normal(0.0, 0.5))
        if decompose_host:
            eta_psf = numpyro.sample('eta_psf_raw', dist.Beta(2.0, 2.0))
        scale_psf = 10.0 ** (-0.4 * delta_m_psf)
        agn_model_psf = scale_psf * agn_model
        gal_model_psf = scale_psf * eta_psf * gal_model_total
        line_model_broad_psf = scale_psf * line_model_broad
        line_model_narrow_psf = scale_psf * eta_psf * line_model_narrow
        line_model_psf = line_model_broad_psf + line_model_narrow_psf
        if line_component_profiles.shape[0] > 0:
            line_component_profiles_psf = line_component_profiles * scale_psf * (
                line_component_broad_mask[:, None]
                + eta_psf * (1.0 - line_component_broad_mask[:, None])
            )
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
        numpyro.deterministic('host_amp', host_amp_out)
        if physical_delayed_host:
            numpyro.deterministic('log_frac_host', log_frac_host)
        numpyro.deterministic('frac_host', frac_host)
    if emit_deterministics and not (fit_pl and fit_reddening):
        numpyro.deterministic('reddening_a2500', reddening_a2500)
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
        numpyro.deterministic('host_aperture_scale', host_aperture_scale)
        numpyro.deterministic('gal_model_intrinsic_total', gal_model_intrinsic_total)
        numpyro.deterministic('gal_model_intrinsic', gal_model_intrinsic)
        numpyro.deterministic('gal_model_total', gal_model_total)
        numpyro.deterministic('gal_model', gal_model)
        numpyro.deterministic('line_model_broad_intrinsic', line_model_broad_intrinsic)
        numpyro.deterministic('line_model_narrow_intrinsic', line_model_narrow_intrinsic)
        numpyro.deterministic('line_model_intrinsic', line_model_intrinsic)
        numpyro.deterministic('line_model_broad', line_model_broad)
        numpyro.deterministic('line_model_narrow', line_model_narrow)
        numpyro.deterministic('line_component_profiles', line_component_profiles)
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
        numpyro.deterministic('line_component_profiles_psf', line_component_profiles_psf)
        numpyro.deterministic('line_model_psf', line_model_psf)
        numpyro.deterministic('psf_model', psf_model)
        for wave_lum, log_lambda_llambda_lum in log_lambda_llambda_agn.items():
            wave_label = _format_wave_label(wave_lum)
            numpyro.deterministic(
                f'log_lambda_Llambda_{wave_label}_agn',
                log_lambda_llambda_lum,
            )
        numpyro.deterministic('fsps_weights', fsps_weights)
        numpyro.deterministic('fsps_weights_frac', fsps_weights_frac)

    student_t_df = float(prior_config.get('student_t_df', 3.0))
    spectral_likelihood_weight = spectral_likelihood_weight_from_resolving_power(
        wave,
        prior_config.get('resolving_power', None),
    )
    if emit_deterministics:
        numpyro.deterministic('spectral_likelihood_weight', spectral_likelihood_weight)
    if has_observed_flux:
        pixel_log_prob = dist.StudentT(df=student_t_df, loc=fiber_model, scale=sigma_tot).log_prob(flux)
        numpyro.factor('obs', spectral_likelihood_weight * jnp.sum(pixel_log_prob))


def quasar_spectral_model(*args, **kwargs):
    """NumPyro spectral model for one configured quasar fit.

    This is the preferred public name for the low-level AGN+host spectral
    model. The historical name :func:`qso_fsps_joint_model` remains available
    for compatibility.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return qso_fsps_joint_model(*args, **kwargs)


def reconstruct_spectral_components(*args, **kwargs):
    """Reconstruct posterior spectral components on a requested wavelength grid.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return reconstruct_posterior_components(*args, **kwargs)


def build_host_template_grid(*args, **kwargs):
    """Build the host-galaxy template grid used by the spectral model.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return build_fsps_template_grid(*args, **kwargs)


def build_tied_line_metadata(*args, **kwargs):
    """Build tied emission-line metadata from a line-list table.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return build_tied_line_meta_from_linelist(*args, **kwargs)


def negative_bal_component(*args, **kwargs):
    """Evaluate a negative Gaussian broad-absorption-line component.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    return negative_gaussian_bal_component(*args, **kwargs)
