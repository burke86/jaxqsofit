"""Typed, unit-explicit public results for standalone spectral fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LineComponentResult:
    """Posterior draws and metadata for one Gaussian line component."""

    amplitude_flambda_1e17: np.ndarray
    center_rest_angstrom: np.ndarray
    sigma_ln_lambda: np.ndarray
    fwhm_kms: np.ndarray
    velocity_offset_kms: np.ndarray
    flux_erg_s_cm2: np.ndarray
    parent_line: str
    component_index: int
    kind: str
    rest_wavelength_angstrom: float


@dataclass(frozen=True)
class LineGroupResult:
    """Aggregate posterior quantities for one physical emission line."""

    component_names: tuple[str, ...]
    total_flux_erg_s_cm2: np.ndarray
    kind: str
    rest_wavelength_angstrom: float


@dataclass(frozen=True)
class SpectralResult:
    """Stable public contract for a standalone jaxqsofit spectrum."""

    lines: Mapping[str, LineComponentResult]
    line_groups: Mapping[str, LineGroupResult]
    wavelength_rest_angstrom: np.ndarray
    observed_flux_flambda_1e17: np.ndarray
    error_flambda_1e17: np.ndarray
    mask: np.ndarray
    model_flambda_1e17: np.ndarray
    continuum_flambda_1e17: np.ndarray
    line_flambda_1e17: np.ndarray
    feii_flambda_1e17: np.ndarray
    balmer_continuum_flambda_1e17: np.ndarray
    host_flambda_1e17: np.ndarray
    power_law_flambda_1e17: np.ndarray


_C_KMS = 299792.458
_GAUSSIAN_FWHM = 2.354820045


def _raw_predictive(fitter: Any) -> Mapping[str, Any]:
    state = fitter._ensure_posterior_state()
    if state.predictive is not None:
        return state.predictive
    return getattr(fitter, "pred_out", {}) or {}


def _interpolate_draws(
    values: Any,
    source_wave: np.ndarray,
    target_wave: np.ndarray,
    n_draws: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[-1] != source_wave.size:
        return np.zeros((n_draws, target_wave.size), dtype=float)
    array = array[:n_draws]
    return np.stack(
        [
            np.interp(target_wave, source_wave, draw, left=0.0, right=0.0)
            for draw in array
        ],
        axis=0,
    )


def _line_results(
    raw: Mapping[str, Any], metadata: Mapping[str, Any], n_draws: int | None
) -> tuple[dict[str, LineComponentResult], dict[str, LineGroupResult]]:
    names = [str(name) for name in metadata.get("names", ())]
    required = (
        "line_amp_per_component",
        "line_mu_per_component",
        "line_sig_per_component",
    )
    if not names or any(name not in raw for name in required):
        return {}, {}
    amplitudes = np.asarray(raw[required[0]], dtype=float)
    centers_ln = np.asarray(raw[required[1]], dtype=float)
    sigmas_ln = np.asarray(raw[required[2]], dtype=float)
    if n_draws is not None:
        amplitudes = amplitudes[:n_draws]
        centers_ln = centers_ln[:n_draws]
        sigmas_ln = sigmas_ln[:n_draws]
    if amplitudes.shape[-1] != len(names):
        raise ValueError(
            "Line metadata does not match the posterior component axis."
        )
    rest_wavelengths = np.asarray(metadata["line_lambda"], dtype=float)
    broad_mask = np.asarray(metadata["broad_mask"], dtype=bool)
    parents = [name.rsplit("_", 1)[0] for name in names]
    parent_counts = {parent: parents.count(parent) for parent in set(parents)}

    lines: dict[str, LineComponentResult] = {}
    members_by_parent: dict[str, list[str]] = {}
    for index, (internal_name, parent) in enumerate(zip(names, parents)):
        public_name = internal_name if parent_counts[parent] > 1 else parent
        amplitude = amplitudes[..., index]
        center_ln = centers_ln[..., index]
        sigma_ln = sigmas_ln[..., index]
        try:
            component_index = int(internal_name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            component_index = len(members_by_parent.get(parent, [])) + 1
        lines[public_name] = LineComponentResult(
            amplitude_flambda_1e17=amplitude,
            center_rest_angstrom=np.exp(center_ln),
            sigma_ln_lambda=sigma_ln,
            fwhm_kms=_GAUSSIAN_FWHM * _C_KMS * sigma_ln,
            velocity_offset_kms=_C_KMS
            * (center_ln - np.log(rest_wavelengths[index])),
            flux_erg_s_cm2=(
                amplitude
                * np.sqrt(2.0 * np.pi)
                * sigma_ln
                * np.exp(center_ln + 0.5 * sigma_ln**2)
                * 1.0e-17
            ),
            parent_line=parent,
            component_index=component_index,
            kind="broad" if broad_mask[index] else "narrow",
            rest_wavelength_angstrom=float(rest_wavelengths[index]),
        )
        members_by_parent.setdefault(parent, []).append(public_name)

    groups: dict[str, LineGroupResult] = {}
    for parent, component_names in members_by_parent.items():
        first = lines[component_names[0]]
        groups[parent] = LineGroupResult(
            component_names=tuple(component_names),
            total_flux_erg_s_cm2=np.sum(
                np.stack(
                    [lines[name].flux_erg_s_cm2 for name in component_names],
                    axis=0,
                ),
                axis=0,
            ),
            kind=first.kind,
            rest_wavelength_angstrom=first.rest_wavelength_angstrom,
        )
    return lines, groups


def build_spectral_result(data: Mapping[str, Any], fitter: Any) -> SpectralResult:
    """Adapt reconstructed and saved posterior arrays to the public contract."""
    wave = np.asarray(data.get("wave", fitter.wave), dtype=float)
    draws = data.get("draws", {})
    raw = _raw_predictive(fitter)
    metadata = getattr(fitter, "tied_line_meta", {}) or {}

    continuum = np.asarray(draws.get("continuum", np.zeros((0, wave.size))), dtype=float)
    if continuum.ndim == 1:
        continuum = continuum[None, :]
    n_draws = continuum.shape[0]
    native_wave = np.asarray(fitter.wave, dtype=float)
    if n_draws == 0 and "continuum_model" in raw:
        n_draws = np.asarray(raw["continuum_model"]).shape[0]
        continuum = _interpolate_draws(
            raw["continuum_model"], native_wave, wave, n_draws
        )
    lines, groups = _line_results(raw, metadata, n_draws)
    line = _interpolate_draws(
        raw.get("line_model", np.zeros((n_draws, native_wave.size))),
        native_wave,
        wave,
        n_draws,
    )
    model = continuum + line
    feii = np.asarray(draws.get("Fe_uv", 0.0), dtype=float) + np.asarray(
        draws.get("Fe_op", 0.0), dtype=float
    )
    if feii.ndim == 0:
        feii = np.zeros_like(continuum)

    same_native_grid = wave.shape == native_wave.shape and np.allclose(
        wave, native_wave, rtol=0.0, atol=1.0e-10
    )
    observed = (
        np.asarray(fitter.flux, dtype=float)
        if same_native_grid
        else np.full(wave.shape, np.nan)
    )
    error = (
        np.asarray(fitter.err, dtype=float)
        if same_native_grid
        else np.full(wave.shape, np.nan)
    )
    mask = (
        np.ones(wave.shape, dtype=bool)
        if same_native_grid
        else np.zeros(wave.shape, dtype=bool)
    )
    return SpectralResult(
        lines=lines,
        line_groups=groups,
        wavelength_rest_angstrom=wave,
        observed_flux_flambda_1e17=observed,
        error_flambda_1e17=error,
        mask=mask,
        model_flambda_1e17=model,
        continuum_flambda_1e17=continuum,
        line_flambda_1e17=line,
        feii_flambda_1e17=feii,
        balmer_continuum_flambda_1e17=np.asarray(
            draws.get("Balmer_cont", np.zeros_like(continuum)), dtype=float
        ),
        host_flambda_1e17=np.asarray(
            draws.get("host", np.zeros_like(continuum)), dtype=float
        ),
        power_law_flambda_1e17=np.asarray(
            draws.get("PL", np.zeros_like(continuum)), dtype=float
        ),
    )


__all__ = [
    "LineComponentResult",
    "LineGroupResult",
    "SpectralResult",
    "build_spectral_result",
]
