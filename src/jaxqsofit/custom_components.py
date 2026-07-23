"""Compatibility alias for custom spectral components."""

import sys

from jaxsedfit import spectral_custom_components as _impl

sys.modules[__name__] = _impl
