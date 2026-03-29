from .core import QSOFit
from .custom_components import (
    CustomComponentSpec,
    CustomLineComponentSpec,
    make_custom_component,
    make_custom_line_component,
    make_template_component,
)
from .model import negative_gaussian_bal_component
from .defaults import (
    DEFAULT_LINE_CONFIG,
    DEFAULT_LINE_PRIOR_ROWS,
    build_default_bal_components,
    build_default_prior_config,
)
from .mplstyle import style_path, use_style

def load_from_samples(*args, **kwargs):
    """Load a saved compressed HDF5 posterior bundle and return a QSOFit object."""
    return QSOFit.load_from_samples(*args, **kwargs)

__all__ = [
    "QSOFit",
    "load_from_samples",
    "CustomComponentSpec",
    "CustomLineComponentSpec",
    "make_custom_component",
    "make_custom_line_component",
    "make_template_component",
    "DEFAULT_LINE_CONFIG",
    "DEFAULT_LINE_PRIOR_ROWS",
    "build_default_bal_components",
    "build_default_prior_config",
    "negative_gaussian_bal_component",
    "style_path",
    "use_style",
]
