from __future__ import annotations

from .config import (
    AGNConfig,
    BALConfig,
    ContinuumConfig,
    ContinuumPriorConfig,
    FeIIPriorConfig,
    HostConfig,
    HostPriorConfig,
    InferenceConfig,
    LineConfig,
    LinePriorConfig,
    Observation,
    OutputConfig,
    PreprocessingConfig,
    PSFPhotometryData,
    PSFPriorConfig,
    FitConfig,
    PriorConfig,
    SpectroscopyData,
    fit_config_from_mapping,
)

__all__ = [
    "JAXQSOFit",
    "FitConfig",
    "FitResult",
    "Observation",
    "SpectroscopyData",
    "PSFPhotometryData",
    "PredictionResult",
    "SpectralResult",
    "LineComponentResult",
    "LineGroupResult",
    "PreprocessingConfig",
    "AGNConfig",
    "BALConfig",
    "ContinuumConfig",
    "ContinuumPriorConfig",
    "FeIIPriorConfig",
    "HostConfig",
    "HostPriorConfig",
    "LineConfig",
    "LinePriorConfig",
    "InferenceConfig",
    "OutputConfig",
    "fit_config_from_mapping",
    "PriorConfig",
    "PSFPriorConfig",
    "load_from_samples",
    "load",
    "CustomComponentSpec",
    "CustomLineComponentSpec",
    "make_custom_component",
    "make_custom_line_component",
    "make_template_component",
    "DEFAULT_LINE_CONFIG",
    "DEFAULT_LINE_PRIOR_ROWS",
    "build_default_bal_components",
    "negative_gaussian_bal_component",
    "SpectralComponentConfig",
    "evaluate_joint_spectral_components",
    "style_path",
    "use_style",
]


def __getattr__(name):
    """Lazily expose model-heavy public objects.

    Parameters
    ----------
    name : object
        name value.
    """
    if name == "JAXQSOFit":
        from .core import JAXQSOFit

        return JAXQSOFit
    if name in {"FitResult", "PredictionResult"}:
        from . import results as _results

        return getattr(_results, name)
    if name in {"SpectralResult", "LineComponentResult", "LineGroupResult"}:
        from . import spectral_results as _spectral_results

        return getattr(_spectral_results, name)
    if name in {"load_from_samples", "load"}:
        from .core import JAXQSOFit

        return JAXQSOFit.load
    if name in {
        "CustomComponentSpec",
        "CustomLineComponentSpec",
        "make_custom_component",
        "make_custom_line_component",
        "make_template_component",
    }:
        from . import custom_components as _custom_components

        return getattr(_custom_components, name)
    if name in {
        "DEFAULT_LINE_CONFIG",
        "DEFAULT_LINE_PRIOR_ROWS",
        "build_default_bal_components",
    }:
        from . import defaults as _defaults

        return getattr(_defaults, name)
    if name == "negative_gaussian_bal_component":
        from .model import negative_gaussian_bal_component

        return negative_gaussian_bal_component
    if name in {"SpectralComponentConfig", "evaluate_joint_spectral_components"}:
        from . import components as _components

        return getattr(_components, name)
    if name in {"style_path", "use_style"}:
        from . import mplstyle as _mplstyle

        return getattr(_mplstyle, name)
    raise AttributeError(name)
