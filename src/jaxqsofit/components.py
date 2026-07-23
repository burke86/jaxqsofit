"""Compatibility alias for reusable spectral components."""

import sys

from jaxsedfit import spectral_components as _impl

sys.modules[__name__] = _impl
