"""Compatibility alias for shared spectral NUTS geometry."""

import sys

from jaxsedfit import spectral_geometry as _impl

sys.modules[__name__] = _impl
