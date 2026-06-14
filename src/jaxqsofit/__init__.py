from __future__ import annotations

from .config import (
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
    "Observation",
    "SpectroscopyData",
    "PSFPhotometryData",
    "PreprocessingConfig",
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
    "build_default_prior_config",
    "quasar_spectral_model",
    "reconstruct_spectral_components",
    "build_host_template_grid",
    "build_tied_line_metadata",
    "negative_bal_component",
    "SpectralComponentConfig",
    "evaluate_joint_spectral_components",
    "style_path",
    "use_style",
]


def __getattr__(name):
    """Lazily expose model-heavy public objects."""
    if name == "JAXQSOFit":
        from .core import JAXQSOFit

        return JAXQSOFit
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
        "build_default_prior_config",
    }:
        from . import defaults as _defaults

        return getattr(_defaults, name)
    if name in {
        "quasar_spectral_model",
        "reconstruct_spectral_components",
        "build_host_template_grid",
        "build_tied_line_metadata",
        "negative_bal_component",
    }:
        from . import model as _model

        return getattr(_model, name)
    if name in {"SpectralComponentConfig", "evaluate_joint_spectral_components"}:
        from . import components as _components

        return getattr(_components, name)
    if name in {"style_path", "use_style"}:
        from . import mplstyle as _mplstyle

        return getattr(_mplstyle, name)
    raise AttributeError(name)
