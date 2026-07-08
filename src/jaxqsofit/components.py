from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .defaults import _build_default_prior_config
from .model import (
    _balmer_continuum_jax,
    _broad_line_mask,
    _fe_template_component,
    _np_to_jnp,
    _sample_tied_line_groups,
    _split_many_gauss_lnlam,
    build_tied_line_meta_from_linelist,
)


@dataclass(frozen=True)
class SpectralComponentConfig:
    """Reusable jaxqsofit spectral-component settings for external joint models.

    ``evaluate_joint_spectral_components`` operates in f_nu units because the
    external SED continuum is passed as ``continuum_mjy``. Internally generated
    Fe II and Balmer-continuum templates are native f_lambda shapes; before they
    are added to the mJy continuum, their shapes are converted to f_nu by
    multiplying by ``(lambda / pivot)^2``. The sampled Fe/Balmer normalizations
    therefore remain mJy-like amplitudes at the configured pivot.
    """

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
    line_coverage_rest: tuple[float, float] | None = None
    line_centers_rest: Sequence[float] | None = None
    line_names: Sequence[str] | None = None
    broad_line_names: Sequence[str] = ()
    line_amp_prior_sigma: float = 2.0
    broad_fwhm_kms_default: float = 3000.0
    narrow_fwhm_kms_default: float = 500.0
    fixed_narrow_fwhm_kms: Any | None = None
    fixed_narrow_amp_scale: Any | None = None
    line_velocity_sigma_kms: float = 500.0
    feii_fwhm_kms_default: float = 3000.0
    balmer_velocity_kms_default: float = 3000.0
    broadening_convolution: str = "fft"
    feii_fnu_pivot_rest: float | None = None
    balmer_fnu_pivot_rest: float | None = 3000.0

    def __post_init__(self) -> None:
        method = str(self.broadening_convolution).lower()
        if method not in {"fft", "direct"}:
            raise ValueError("SpectralComponentConfig.broadening_convolution must be 'fft' or 'direct'.")
        object.__setattr__(self, "broadening_convolution", method)


def _as_config(config: SpectralComponentConfig | None) -> SpectralComponentConfig:
    """Return an explicit spectral-component config, filling defaults when absent.


    Parameters
    ----------
    config : object
        config value.
    """
    return config if isinstance(config, SpectralComponentConfig) else SpectralComponentConfig()


def _component_prior_config(cfg: SpectralComponentConfig) -> dict[str, Any]:
    """Return a jaxqsofit-style prior config for external component fits.

    Parameters
    ----------
    cfg : object
        cfg value.
    """
    if cfg.line_prior_config is None:
        prior = _build_default_prior_config(
            np.asarray([max(float(cfg.line_flux_scale_mjy), 1.0e-8)], dtype=float),
            include_elg_narrow_lines=bool(cfg.include_elg_narrow_lines),
            include_high_ionization_lines=bool(cfg.include_high_ionization_lines),
        )
        if hasattr(prior, "to_mapping"):
            prior = prior.to_mapping()
    else:
        prior = cfg.line_prior_config
        if hasattr(prior, "to_mapping"):
            prior = prior.to_mapping()
        prior = copy.deepcopy(dict(prior))
    if cfg.line_table is not None:
        prior.setdefault("line", {})
        prior["line"]["table"] = [dict(row) for row in cfg.line_table]
    return prior


def _line_table_from_prior_config(prior_config: Mapping[str, Any]):
    """Extract a line table from the canonical ``line.table`` layout.

    Parameters
    ----------
    prior_config : object
        prior_config value.
    """
    line_cfg = prior_config.get("line", None)
    if isinstance(line_cfg, Mapping):
        if "table" in line_cfg:
            return line_cfg["table"]
    return None


def _filter_line_table_to_rest_coverage(
    line_table: Sequence[Mapping[str, Any]],
    coverage_rest: tuple[float, float] | None,
) -> list[Mapping[str, Any]]:
    """Keep only line-table rows whose fitting window overlaps rest coverage.

    Parameters
    ----------
    line_table : object
        line_table value.
    coverage_rest : object
        coverage_rest value.
    """
    if coverage_rest is None:
        return [dict(row) for row in line_table]
    lo, hi = sorted((float(coverage_rest[0]), float(coverage_rest[1])))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    covered: list[Mapping[str, Any]] = []
    for row in line_table:
        lam = float(row.get("lambda", np.nan))
        row_lo = float(row.get("minwav", lam))
        row_hi = float(row.get("maxwav", lam))
        if np.isfinite(row_lo) and np.isfinite(row_hi) and max(row_lo, lo) <= min(row_hi, hi):
            covered.append(dict(row))
    return covered


def _evaluate_tied_line_components(wave_rest, cfg: SpectralComponentConfig, *, site_prefix: str):
    """Evaluate jaxqsofit's grouped tied-line model on a rest-frame grid.

    Parameters
    ----------
    wave_rest : object
        wave_rest value.
    cfg : object
        cfg value.
    site_prefix : object
        site_prefix value.
    """
    prior_config = _component_prior_config(cfg)
    line_table = _line_table_from_prior_config(prior_config)
    if line_table is None:
        return jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), {}
    line_table = _filter_line_table_to_rest_coverage(line_table, cfg.line_coverage_rest)

    tied_line_meta = build_tied_line_meta_from_linelist(
        line_table,
        np.asarray([1.0, 1.0e8], dtype=float),
    )
    if int(tied_line_meta["n_lines"]) <= 0:
        return jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), jnp.zeros_like(wave_rest), {}

    dmu_group, sig_group, amp_group = _sample_tied_line_groups(
        tied_line_meta,
        prior_config,
        site_prefix=site_prefix,
    )

    dmu = dmu_group[jnp.asarray(tied_line_meta["vgroup"], dtype=jnp.int32)]
    sigs = sig_group[jnp.asarray(tied_line_meta["wgroup"], dtype=jnp.int32)]
    amps = amp_group[jnp.asarray(tied_line_meta["fgroup"], dtype=jnp.int32)] * jnp.asarray(tied_line_meta["flux_ratio"], dtype=jnp.float64)
    mus = jnp.asarray(tied_line_meta["ln_lambda0"], dtype=jnp.float64) + dmu

    broad_mask = jnp.asarray(_broad_line_mask(tied_line_meta.get("names", [])), dtype=jnp.float64)
    if cfg.fixed_narrow_fwhm_kms is not None:
        fixed_narrow_sig = jnp.maximum(
            jnp.asarray(cfg.fixed_narrow_fwhm_kms, dtype=jnp.float64),
            1.0,
        ) / (299792.458 * 2.354820045)
        sigs = jnp.where(broad_mask > 0.0, sigs, fixed_narrow_sig)
    narrow_amp_scale = (
        jnp.maximum(jnp.asarray(cfg.fixed_narrow_amp_scale, dtype=jnp.float64), 1.0e-12)
        if cfg.fixed_narrow_amp_scale is not None
        else jnp.asarray(1.0, dtype=jnp.float64)
    )
    amps = amps * (broad_mask + (1.0 - broad_mask) * narrow_amp_scale)
    narrow_weights = jnp.clip(amps * (1.0 - broad_mask), 0.0, None)
    narrow_weight_sum = jnp.sum(narrow_weights)
    narrow_fwhm_kms = jnp.where(
        narrow_weight_sum > 0.0,
        299792.458 * 2.354820045 * jnp.sum(sigs * narrow_weights) / jnp.maximum(narrow_weight_sum, 1.0e-30),
        jnp.asarray(float(cfg.narrow_fwhm_kms_default), dtype=jnp.float64),
    )
    lnwave = jnp.log(wave_rest)
    total, broad, narrow, _ = _split_many_gauss_lnlam(
        lnwave,
        amps,
        mus,
        sigs,
        broad_mask,
    )
    diagnostics = {
        "line_amp_per_component": amps,
        "line_mu_per_component": mus,
        "line_sig_per_component": sigs,
        "line_narrow_fwhm_kms": narrow_fwhm_kms,
        "line_narrow_amp_scale": narrow_amp_scale,
    }
    return total, broad, narrow, diagnostics


def _evaluate_simple_line_components(wave_rest, continuum_model, cfg: SpectralComponentConfig, *, site_prefix: str):
    """Backward-compatible explicit Gaussian line list.

    Parameters
    ----------
    wave_rest : object
        wave_rest value.
    continuum_model : object
        continuum_model value.
    cfg : object
        cfg value.
    site_prefix : object
        site_prefix value.
    """
    line_model = jnp.zeros_like(wave_rest)
    broad_model = jnp.zeros_like(wave_rest)
    narrow_model = jnp.zeros_like(wave_rest)
    if not cfg.line_centers_rest:
        return line_model, broad_model, narrow_model, {}
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
        if (not is_broad) and cfg.fixed_narrow_fwhm_kms is not None:
            fwhm = jnp.maximum(jnp.asarray(cfg.fixed_narrow_fwhm_kms, dtype=jnp.float64), 1.0)
        if (not is_broad) and cfg.fixed_narrow_amp_scale is not None:
            amp = amp * jnp.maximum(jnp.asarray(cfg.fixed_narrow_amp_scale, dtype=jnp.float64), 1.0e-12)
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
    return line_model, broad_model, narrow_model, {
        "line_narrow_fwhm_kms": (
            jnp.maximum(jnp.asarray(cfg.fixed_narrow_fwhm_kms, dtype=jnp.float64), 1.0)
            if cfg.fixed_narrow_fwhm_kms is not None
            else jnp.asarray(float(cfg.narrow_fwhm_kms_default), dtype=jnp.float64)
        ),
        "line_narrow_amp_scale": (
            jnp.maximum(jnp.asarray(cfg.fixed_narrow_amp_scale, dtype=jnp.float64), 1.0e-12)
            if cfg.fixed_narrow_amp_scale is not None
            else jnp.asarray(1.0, dtype=jnp.float64)
        ),
    }


def _flambda_shape_to_fnu_mjy_shape(wave_rest, flambda_shape, pivot_rest):
    """Convert a relative f_lambda component shape to an f_nu shape.

    The conversion is normalized at ``pivot_rest`` so component amplitudes stay
    in the same mJy-like scale. This preserves the external API while avoiding
    adding f_lambda-shaped Fe/Balmer templates directly to an f_nu continuum.

    Parameters
    ----------
    wave_rest : object
        wave_rest value.
    flambda_shape : object
        flambda_shape value.
    pivot_rest : object
        pivot_rest value.
    """
    wave_rest = jnp.asarray(wave_rest, dtype=jnp.float64)
    flambda_shape = jnp.asarray(flambda_shape, dtype=jnp.float64)
    if pivot_rest is None:
        pivot = jnp.nanmedian(wave_rest)
    else:
        pivot = jnp.asarray(float(pivot_rest), dtype=jnp.float64)
    pivot = jnp.maximum(pivot, 1.0e-8)
    return flambda_shape * jnp.square(jnp.clip(wave_rest, 1.0e-8, None) / pivot)


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
    feii_template_wave_rest, feii_template_flux
        Rest-frame Fe II template sampled as an f_lambda-shaped relative
        spectrum. The evaluated Fe II component is converted to f_nu shape
        before being added to the mJy continuum.

    config : object
        config value.

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
            line_model, line_broad_model, line_narrow_model, line_diagnostics = _evaluate_simple_line_components(
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
        feii_flambda_shape = _fe_template_component(
            wave_rest,
            jnp.asarray(np.asarray(feii_template_wave_rest, dtype=float)),
            jnp.asarray(np.asarray(feii_template_flux, dtype=float)),
            feii_norm,
            feii_fwhm,
            feii_shift,
            convolution_method=cfg.broadening_convolution,
        )
        feii_model = _flambda_shape_to_fnu_mjy_shape(
            wave_rest,
            feii_flambda_shape,
            cfg.feii_fnu_pivot_rest,
        )

    balmer_model = jnp.zeros_like(wave_obs)
    if cfg.use_balmer_continuum:
        balmer_norm = numpyro.sample(f"{site_prefix}_balmer_norm", dist.LogNormal(jnp.log(1.0e-3), 2.0))
        balmer_tau = numpyro.sample(f"{site_prefix}_balmer_tau", dist.LogNormal(jnp.log(1.0), 0.5))
        balmer_vel = numpyro.sample(
            f"{site_prefix}_balmer_vel",
            dist.LogNormal(jnp.log(max(cfg.balmer_velocity_kms_default, 1.0)), 0.4),
        )
        balmer_flambda_shape = _balmer_continuum_jax(
            wave_rest,
            balmer_norm,
            15000.0,
            balmer_tau,
            balmer_vel,
            convolution_method=cfg.broadening_convolution,
        )
        balmer_model = _flambda_shape_to_fnu_mjy_shape(
            wave_rest,
            balmer_flambda_shape,
            cfg.balmer_fnu_pivot_rest,
        )

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
