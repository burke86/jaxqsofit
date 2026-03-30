from __future__ import annotations

import copy
import importlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np


def _sanitize_component_name(name: str) -> str:
    """Return a stable ASCII-safe identifier for a custom component."""
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        raise ValueError("Custom component name must contain at least one alphanumeric character.")
    if text[0].isdigit():
        text = f"c_{text}"
    return text


def _callable_to_ref(func: Callable[..., Any]) -> str:
    """Serialize an importable top-level function to module:qualname."""
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(
            "Custom component evaluators must be importable top-level functions "
            "to support save/load of posterior bundles."
        )
    return f"{module}:{qualname}"


def _callable_from_ref(ref: str) -> Callable[..., Any]:
    """Deserialize a module:qualname function reference."""
    module_name, qualname = str(ref).split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"Resolved custom component reference is not callable: {ref}")
    return obj


@dataclass(frozen=True)
class CustomComponentSpec:
    """Generic additive continuum component.

    The component is fully defined by:
    - ``parameter_priors``: local parameter names -> prior config dictionaries
    - ``evaluate``: callable ``evaluate(wave, params, metadata)``

    The evaluator is responsible for any shifts, broadenings, template
    interpolation, or other transformations.
    """

    name: str
    parameter_priors: Mapping[str, Mapping[str, Any]]
    evaluate: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        safe_name = _sanitize_component_name(self.name)
        object.__setattr__(self, "name", safe_name)
        if not callable(self.evaluate):
            raise TypeError("Custom component evaluate must be callable.")
        priors = {str(k): dict(v) for k, v in dict(self.parameter_priors).items()}
        object.__setattr__(self, "parameter_priors", priors)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def prefix(self) -> str:
        """Return the parameter-site prefix used in samples/priors."""
        return f"custom_{self.name}"

    @property
    def output_name(self) -> str:
        """Return the public output component key."""
        return self.name

    @property
    def deterministic_site_name(self) -> str:
        """Return the Predictive deterministic site name."""
        return f"{self.prefix}_model"

    def site_name(self, param_name: str) -> str:
        """Return the full NumPyro sample-site name for one local parameter."""
        return f"{self.prefix}_{param_name}"

    def to_state(self) -> dict[str, Any]:
        """Return a pickle-friendly representation."""
        return {
            "__custom_component__": True,
            "name": self.name,
            "parameter_priors": copy.deepcopy(dict(self.parameter_priors)),
            "evaluate_ref": _callable_to_ref(self.evaluate),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CustomComponentSpec":
        """Rebuild a spec from :meth:`to_state`."""
        return cls(
            name=str(state["name"]),
            parameter_priors=dict(state["parameter_priors"]),
            evaluate=_callable_from_ref(str(state["evaluate_ref"])),
            metadata=dict(state.get("metadata", {})),
        )


@dataclass(frozen=True)
class CustomLineComponentSpec:
    """Generic additive emission-line component."""

    name: str
    parameter_priors: Mapping[str, Mapping[str, Any]]
    evaluate: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any]
    line_kind: str = "broad"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        safe_name = _sanitize_component_name(self.name)
        object.__setattr__(self, "name", safe_name)
        if not callable(self.evaluate):
            raise TypeError("Custom line component evaluate must be callable.")
        priors = {str(k): dict(v) for k, v in dict(self.parameter_priors).items()}
        object.__setattr__(self, "parameter_priors", priors)
        kind = str(self.line_kind).strip().lower()
        if kind not in {"broad", "narrow"}:
            raise ValueError("Custom line component line_kind must be 'broad' or 'narrow'.")
        object.__setattr__(self, "line_kind", kind)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def prefix(self) -> str:
        return f"custom_line_{self.name}"

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def deterministic_site_name(self) -> str:
        return f"{self.prefix}_model"

    def site_name(self, param_name: str) -> str:
        return f"{self.prefix}_{param_name}"

    def to_state(self) -> dict[str, Any]:
        return {
            "__custom_line_component__": True,
            "name": self.name,
            "parameter_priors": copy.deepcopy(dict(self.parameter_priors)),
            "evaluate_ref": _callable_to_ref(self.evaluate),
            "line_kind": self.line_kind,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CustomLineComponentSpec":
        return cls(
            name=str(state["name"]),
            parameter_priors=dict(state["parameter_priors"]),
            evaluate=_callable_from_ref(str(state["evaluate_ref"])),
            line_kind=str(state.get("line_kind", "broad")),
            metadata=dict(state.get("metadata", {})),
        )


def make_custom_component(
    name: str,
    parameter_priors: Mapping[str, Mapping[str, Any]],
    evaluate: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CustomComponentSpec:
    """Build a generic additive custom component."""
    return CustomComponentSpec(
        name=name,
        parameter_priors=parameter_priors,
        evaluate=evaluate,
        metadata={} if metadata is None else dict(metadata),
    )


def make_custom_line_component(
    name: str,
    parameter_priors: Mapping[str, Mapping[str, Any]],
    evaluate: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any],
    *,
    line_kind: str = "broad",
    metadata: Mapping[str, Any] | None = None,
) -> CustomLineComponentSpec:
    """Build a generic additive custom line component."""
    return CustomLineComponentSpec(
        name=name,
        parameter_priors=parameter_priors,
        evaluate=evaluate,
        line_kind=line_kind,
        metadata={} if metadata is None else dict(metadata),
    )


def template_component_evaluator(wave, params, metadata):
    """Convenience evaluator for broadened/shifted template components."""
    from .model import _fe_template_component

    wave_template = jnp.asarray(metadata["wave"], dtype=jnp.float64)
    flux_template = jnp.asarray(
        metadata.get("flux_model", metadata["flux"]),
        dtype=jnp.float64,
    )
    return _fe_template_component(
        wave,
        wave_template,
        flux_template,
        params["norm"],
        params.get("fwhm", jnp.asarray(metadata.get("default_fwhm_kms", 3000.0), dtype=jnp.float64)),
        params.get("shift", 0.0),
        base_fwhm_kms=float(metadata.get("base_fwhm_kms", 900.0)),
    )


def make_template_component(
    name: str,
    wave: Sequence[float],
    flux: Sequence[float],
    *,
    fit_fwhm: bool = False,
    fit_shift: bool = False,
    base_fwhm_kms: float = 900.0,
    default_fwhm_kms: float = 3000.0,
    normalize_template: bool = True,
    target_amp: float = 1.0,
) -> CustomComponentSpec:
    """Build a broadened/shifted additive template component."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if wave.ndim != 1 or flux.ndim != 1 or wave.size != flux.size or wave.size < 2:
        raise ValueError("Template custom component wave/flux must be same-length 1D arrays.")
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(flux)):
        raise ValueError("Template custom component wave/flux must be finite.")
    if np.any(np.diff(wave) <= 0):
        raise ValueError("Template custom component wavelength grid must be strictly increasing.")

    priors: dict[str, dict[str, Any]] = {
        "norm": {"dist": "LogNormal", "loc": np.log(1e-3), "scale": 0.5},
    }
    if fit_fwhm:
        priors["fwhm"] = {"dist": "LogNormal", "loc": np.log(float(default_fwhm_kms)), "scale": 0.3}
    if fit_shift:
        priors["shift"] = {"dist": "Normal", "loc": 0.0, "scale": 1e-3}

    flux_model = flux
    if bool(normalize_template):
        flux_model = _normalize_template_flux(flux_model, target_amp=float(target_amp))

    return make_custom_component(
        name=name,
        parameter_priors=priors,
        evaluate=template_component_evaluator,
        metadata={
            "wave": wave,
            "flux": flux,
            "flux_model": flux_model,
            "normalize_template": bool(normalize_template),
            "target_amp": float(target_amp),
            "base_fwhm_kms": float(base_fwhm_kms),
            "default_fwhm_kms": float(default_fwhm_kms),
        },
    )


def normalize_custom_components(custom_components: Iterable[CustomComponentSpec] | None) -> tuple[CustomComponentSpec, ...]:
    """Validate and normalize custom component definitions."""
    if custom_components is None:
        return ()
    normalized = []
    seen = set()
    for comp in custom_components:
        if not isinstance(comp, CustomComponentSpec):
            raise TypeError("custom_components entries must be CustomComponentSpec objects.")
        if comp.name in seen:
            raise ValueError(f"Duplicate custom component name: {comp.name}")
        seen.add(comp.name)
        normalized.append(comp)
    return tuple(normalized)


def normalize_custom_line_components(
    custom_line_components: Iterable[CustomLineComponentSpec] | None,
) -> tuple[CustomLineComponentSpec, ...]:
    """Validate and normalize custom line-component definitions."""
    if custom_line_components is None:
        return ()
    normalized = []
    seen = set()
    for comp in custom_line_components:
        if not isinstance(comp, CustomLineComponentSpec):
            raise TypeError("custom_line_components entries must be CustomLineComponentSpec objects.")
        if comp.name in seen:
            raise ValueError(f"Duplicate custom line component name: {comp.name}")
        seen.add(comp.name)
        normalized.append(comp)
    return tuple(normalized)


def inject_default_custom_component_priors(
    prior_config: dict[str, Any],
    flux: np.ndarray,
    custom_components: Iterable[CustomComponentSpec] | None,
) -> dict[str, Any]:
    """Return a prior config with default keys added for custom components."""
    comps = normalize_custom_components(custom_components)
    if len(comps) == 0:
        return prior_config

    cfg = copy.deepcopy(prior_config)
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    if (not np.isfinite(fscale)) or fscale <= 0:
        fscale = 1.0

    for comp in comps:
        for param_name, param_cfg in comp.parameter_priors.items():
            out_cfg = dict(param_cfg)
            scale = float(out_cfg.get("scale", 1.0))
            loc = float(out_cfg.get("loc", 0.0))
            if str(out_cfg.get("dist", "Normal")).lower() == "lognormal" and loc <= np.log(1e-8):
                out_cfg["loc"] = np.log(max(1e-3 * fscale, 1e-10))
            elif str(param_name) == "c0" and scale == 0.0:
                out_cfg["scale"] = 0.2 * fscale
            elif str(param_name).startswith("c") and scale == 0.0:
                out_cfg["scale"] = 0.05 * fscale
            cfg.setdefault(comp.site_name(param_name), out_cfg)
    return cfg


def inject_default_custom_line_component_priors(
    prior_config: dict[str, Any],
    flux: np.ndarray,
    custom_line_components: Iterable[CustomLineComponentSpec] | None,
) -> dict[str, Any]:
    """Return a prior config with default keys added for custom line components."""
    comps = normalize_custom_line_components(custom_line_components)
    if len(comps) == 0:
        return prior_config

    cfg = copy.deepcopy(prior_config)
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    if (not np.isfinite(fscale)) or fscale <= 0:
        fscale = 1.0

    for comp in comps:
        for param_name, param_cfg in comp.parameter_priors.items():
            out_cfg = dict(param_cfg)
            scale = float(out_cfg.get("scale", 1.0))
            loc = float(out_cfg.get("loc", 0.0))
            if str(out_cfg.get("dist", "Normal")).lower() == "lognormal" and loc <= np.log(1e-8):
                out_cfg["loc"] = np.log(max(1e-3 * fscale, 1e-10))
            elif scale == 0.0:
                out_cfg["scale"] = 0.05 * fscale
            cfg.setdefault(comp.site_name(param_name), out_cfg)
    return cfg


def custom_component_site_names(custom_components: Iterable[CustomComponentSpec] | None) -> list[str]:
    """Return deterministic-site names used by Predictive for custom components."""
    return [comp.deterministic_site_name for comp in normalize_custom_components(custom_components)]


def custom_line_component_site_names(
    custom_line_components: Iterable[CustomLineComponentSpec] | None,
) -> list[str]:
    """Return deterministic-site names used by Predictive for custom line components."""
    return [comp.deterministic_site_name for comp in normalize_custom_line_components(custom_line_components)]
