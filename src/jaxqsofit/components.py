from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .defaults import build_default_prior_config
from .model import (
    _balmer_continuum_jax,
    _broad_line_mask,
    _fe_template_component,
    _many_gauss_lnlam,
    _np_to_jnp,
    build_tied_line_meta_from_linelist,
)


@dataclass(frozen=True)
class SpectralComponentConfig:
    """Reusable jaxqsofit spectral-component settings for external joint models."""

    use_lines: bool = True
    use_feii: bool = False
    use_balmer_continuum: bool = False
    use_multiplicative_tilt: bool = False
    use_tied_lines: bool = True
    line_table: Sequence[Mapping[str, Any]] | None = None
    line_prior_config: Mapping[str, Any] | None = None
    line_flux_scale_mjy: float = 1.0
    include_elg_narrow_lines: bool = False
    include_high_ionization_lines: bool = False
    line_centers_rest: Sequence[float] | None = None
    line_names: Sequence[str] | None = None
    broad_line_names: Sequence[str] = ()
    line_amp_prior_sigma: float = 2.0
    broad_fwhm_kms_default: float = 3000.0
    narrow_fwhm_kms_default: float = 500.0
    line_velocity_sigma_kms: float = 500.0
    feii_fwhm_kms_default: float = 3000.0
    balmer_velocity_kms_default: float = 3000.0


def _as_config(config: SpectralComponentConfig | None) -> SpectralComponentConfig:
    return config if isinstance(config, SpectralComponentConfig) else SpectralComponentConfig()


def _component_prior_config(cfg: SpectralComponentConfig) -> dict[str, Any]:
    """Return a jaxqsofit-style prior config for external component fits."""
    if cfg.line_prior_config is None:
        prior = build_default_prior_config(
            np.asarray([max(float(cfg.line_flux_scale_mjy), 1.0e-8)], dtype=float),
            include_elg_narrow_lines=bool(cfg.include_elg_narrow_lines),
            include_high_ionization_lines=bool(cfg.include_high_ionization_lines),
        )
    else:
        prior = copy.deepcopy(dict(cfg.line_prior_config))
    if cfg.line_table is not None:
        prior.setdefault("line", {})
        prior["line"]["table"] = [dict(row) for row in cfg.line_table]
    return prior


def _line_table_from_prior_config(prior_config: Mapping[str, Any]):
    line_cfg = prior_config.get("line", None)
    if isinstance(line_cfg, Mapping):
        if "table" in line_cfg:
            return line_cfg["table"]
        if "priors" in line_cfg:
            return line_cfg["priors"]
    if "line_table" in prior_config:
        return prior_config["line_table"]
    if "line_priors" in prior_config:
        return prior_config["line_priors"]
    return None


def _evaluate_tied_line_components(wave_rest, cfg: SpectralComponentConfig, *, site_prefix: str):
    """Evaluate jaxqsofit's grouped tied-line model on a rest-frame grid."""
    prior_config = _component_prior_config(cfg)
    line_table = _line_table_from_prior_config(prior_config)
    if line_table is None:
        return jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), {}

    # Include all configured lines. Lines outside the current spectral window
    # evaluate to negligible flux but keeping the full table preserves ties.
    tied_line_meta = build_tied_line_meta_from_linelist(
        line_table,
        np.asarray([1.0, 1.0e8], dtype=float),
    )
    if int(tied_line_meta["n_lines"]) <= 0:
        return jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), {}

    n_v = int(tied_line_meta["n_vgroups"])
    n_w = int(tied_line_meta["n_wgroups"])
    n_f = int(tied_line_meta["n_fgroups"])
    dmu_scale_mult = float(prior_config.get("line_dmu_scale_mult", 0.25))
    sig_scale_mult = float(prior_config.get("line_sig_scale_mult", 0.25))
    amp_scale_mult = float(prior_config.get("line_amp_scale_mult", 0.25))

    dmu_group = numpyro.sample(
        f"{site_prefix}_line_dmu_group",
        dist.TruncatedNormal(
            loc=_np_to_jnp(tied_line_meta["dmu_init_group"]),
            scale=_np_to_jnp(np.maximum(dmu_scale_mult * (tied_line_meta["dmu_max_group"] - tied_line_meta["dmu_min_group"]), 1.0e-6)),
            low=_np_to_jnp(tied_line_meta["dmu_min_group"]),
            high=_np_to_jnp(tied_line_meta["dmu_max_group"]),
        ),
    ) if n_v > 0 else jnp.zeros((0,))

    sig_group = numpyro.sample(
        f"{site_prefix}_line_sig_group",
        dist.TruncatedNormal(
            loc=_np_to_jnp(np.clip(tied_line_meta["sig_init_group"], 1.0e-5, None)),
            scale=_np_to_jnp(np.maximum(sig_scale_mult * (tied_line_meta["sig_max_group"] - tied_line_meta["sig_min_group"]), 1.0e-6)),
            low=_np_to_jnp(np.clip(tied_line_meta["sig_min_group"], 1.0e-5, None)),
            high=_np_to_jnp(np.clip(tied_line_meta["sig_max_group"], 1.0e-5, None)),
        ),
    ) if n_w > 0 else jnp.zeros((0,))

    amp_group = numpyro.sample(
        f"{site_prefix}_line_amp_group",
        dist.TruncatedNormal(
            loc=_np_to_jnp(np.clip(tied_line_meta["amp_init_group"], 1.0e-10, None)),
            scale=_np_to_jnp(np.maximum(amp_scale_mult * (tied_line_meta["amp_max_group"] - tied_line_meta["amp_min_group"]), 1.0e-10)),
            low=_np_to_jnp(np.clip(tied_line_meta["amp_min_group"], 1.0e-10, None)),
            high=_np_to_jnp(np.clip(tied_line_meta["amp_max_group"], 1.0e-10, None)),
        ),
    ) if n_f > 0 else jnp.zeros((0,))

    dmu = dmu_group[jnp.asarray(tied_line_meta["vgroup"], dtype=jnp.int32)]
    sigs = sig_group[jnp.asarray(tied_line_meta["wgroup"], dtype=jnp.int32)]
    amps = amp_group[jnp.asarray(tied_line_meta["fgroup"], dtype=jnp.int32)] * jnp.asarray(tied_line_meta["flux_ratio"], dtype=jnp.float64)
    mus = jnp.asarray(tied_line_meta["ln_lambda0"], dtype=jnp.float64) + dmu

    broad_mask = jnp.asarray(_broad_line_mask(tied_line_meta.get("names", [])), dtype=jnp.float64)
    lnwave = jnp.log(wave_rest)
    broad = _many_gauss_lnlam(lnwave, amps * broad_mask, mus, sigs)
    narrow = _many_gauss_lnlam(lnwave, amps * (1.0 - broad_mask), mus, sigs)
    total = broad + narrow
    diagnostics = {
        "line_amp_per_component": amps,
        "line_mu_per_component": mus,
        "line_sig_per_component": sigs,
    }
    return total, broad, narrow, diagnostics


def _evaluate_simple_line_components(wave_rest, continuum_model, cfg: SpectralComponentConfig, *, site_prefix: str):
    """Backward-compatible explicit Gaussian line list."""
    line_model = jnp.zeros_like(wave_rest)
    broad_model = jnp.zeros_like(wave_rest)
    narrow_model = jnp.zeros_like(wave_rest)
    if not cfg.line_centers_rest:
        return line_model, broad_model, narrow_model
    line_names = cfg.line_names or tuple(f"line_{i}" for i, _ in enumerate(cfg.line_centers_rest))
    broad_names = {str(name) for name in cfg.broad_line_names}
    cont_scale = jnp.maximum(jnp.nanmedian(jnp.abs(continuum_model)), 1.0e-8)
    for name, center in zip(line_names, cfg.line_centers_rest):
        is_broad = str(name) in broad_names
        default_fwhm = cfg.broad_fwhm_kms_default if is_broad else cfg.narrow_fwhm_kms_default
        amp = numpyro.sample(
            f"{site_prefix}_line_amp_{name}",
            dist.LogNormal(jnp.log(cont_scale * 0.1), cfg.line_amp_prior_sigma),
        )
        fwhm = numpyro.sample(
            f"{site_prefix}_line_fwhm_{name}",
            dist.LogNormal(jnp.log(max(default_fwhm, 1.0)), 0.5),
        )
        velocity = numpyro.sample(
            f"{site_prefix}_line_velocity_{name}",
            dist.Normal(0.0, max(cfg.line_velocity_sigma_kms, 1.0)),
        )
        center_shifted = jnp.asarray(float(center), dtype=jnp.float64) * (1.0 + velocity / 299792.458)
        sigma = jnp.maximum(center_shifted * fwhm / 299792.458 / 2.354820045, 1.0e-6)
        component = amp * jnp.exp(-0.5 * jnp.square((wave_rest - center_shifted) / sigma))
        line_model = line_model + component
        broad_model = broad_model + jnp.where(is_broad, component, 0.0)
        narrow_model = narrow_model + jnp.where(is_broad, 0.0, component)
    return line_model, broad_model, narrow_model


def evaluate_joint_spectral_components(
    wave_obs,
    redshift,
    continuum_mjy,
    *,
    config: SpectralComponentConfig | None = None,
    feii_template_wave_rest=None,
    feii_template_flux=None,
    site_prefix: str = "jqf",
):
    """Evaluate jaxqsofit spectral components around an external continuum.

    Parameters
    ----------
    wave_obs
        Observed-frame wavelength grid in Angstrom.
    redshift
        Source redshift.
    continuum_mjy
        External continuum prediction on ``wave_obs`` in mJy. In a joint
        grahspj fit this is the shared AGN+host continuum.

    Returns
    -------
    dict
        JAX arrays for total model and individual component contributions in
        mJy. The function samples NumPyro parameters with names prefixed by
        ``site_prefix`` so it can run inside a larger joint model.
    """
    cfg = _as_config(config)
    wave_obs = jnp.asarray(wave_obs, dtype=jnp.float64)
    continuum_mjy = jnp.asarray(continuum_mjy, dtype=jnp.float64)
    redshift = jnp.maximum(jnp.asarray(redshift, dtype=jnp.float64), 0.0)
    wave_rest = wave_obs / jnp.maximum(1.0 + redshift, 1.0e-8)

    calibration = jnp.ones_like(wave_obs)
    if cfg.use_multiplicative_tilt:
        tilt = numpyro.sample(f"{site_prefix}_continuum_tilt", dist.Normal(0.0, 0.1))
        pivot = jnp.maximum(jnp.nanmedian(wave_obs), 1.0)
        calibration = jnp.clip((wave_obs / pivot) ** tilt, 0.2, 5.0)

    continuum_model = calibration * continuum_mjy
    line_model = jnp.zeros_like(wave_obs)
    line_broad_model = jnp.zeros_like(wave_obs)
    line_narrow_model = jnp.zeros_like(wave_obs)
    line_diagnostics = {}
    if cfg.use_lines:
        if cfg.use_tied_lines and cfg.line_centers_rest is None:
            line_model, line_broad_model, line_narrow_model, line_diagnostics = _evaluate_tied_line_components(
                wave_rest,
                cfg,
                site_prefix=site_prefix,
            )
        else:
            line_model, line_broad_model, line_narrow_model = _evaluate_simple_line_components(
                wave_rest,
                continuum_model,
                cfg,
                site_prefix=site_prefix,
            )

    feii_model = jnp.zeros_like(wave_obs)
    if cfg.use_feii and feii_template_wave_rest is not None and feii_template_flux is not None:
        feii_norm = numpyro.sample(f"{site_prefix}_feii_norm", dist.LogNormal(jnp.log(1.0e-3), 2.0))
        feii_fwhm = numpyro.sample(
            f"{site_prefix}_feii_fwhm",
            dist.LogNormal(jnp.log(max(cfg.feii_fwhm_kms_default, 1.0)), 0.4),
        )
        feii_shift = numpyro.sample(f"{site_prefix}_feii_shift", dist.Normal(0.0, 0.01))
        feii_model = _fe_template_component(
            wave_rest,
            jnp.asarray(np.asarray(feii_template_wave_rest, dtype=float)),
            jnp.asarray(np.asarray(feii_template_flux, dtype=float)),
            feii_norm,
            feii_fwhm,
            feii_shift,
        )

    balmer_model = jnp.zeros_like(wave_obs)
    if cfg.use_balmer_continuum:
        balmer_norm = numpyro.sample(f"{site_prefix}_balmer_norm", dist.LogNormal(jnp.log(1.0e-3), 2.0))
        balmer_tau = numpyro.sample(f"{site_prefix}_balmer_tau", dist.LogNormal(jnp.log(1.0), 0.5))
        balmer_vel = numpyro.sample(
            f"{site_prefix}_balmer_vel",
            dist.LogNormal(jnp.log(max(cfg.balmer_velocity_kms_default, 1.0)), 0.4),
        )
        balmer_model = _balmer_continuum_jax(wave_rest, balmer_norm, 15000.0, balmer_tau, balmer_vel)

    total = continuum_model + line_model + feii_model + balmer_model
    numpyro.deterministic(f"{site_prefix}_continuum_model", continuum_model)
    numpyro.deterministic(f"{site_prefix}_line_model", line_model)
    numpyro.deterministic(f"{site_prefix}_line_model_broad", line_broad_model)
    numpyro.deterministic(f"{site_prefix}_line_model_narrow", line_narrow_model)
    for name, value in line_diagnostics.items():
        numpyro.deterministic(f"{site_prefix}_{name}", value)
    numpyro.deterministic(f"{site_prefix}_feii_model", feii_model)
    numpyro.deterministic(f"{site_prefix}_balmer_model", balmer_model)
    numpyro.deterministic(f"{site_prefix}_total_model", total)
    return {
        "total": total,
        "continuum": continuum_model,
        "lines": line_model,
        "line_broad": line_broad_model,
        "line_narrow": line_narrow_model,
        "feii": feii_model,
        "balmer": balmer_model,
    }
