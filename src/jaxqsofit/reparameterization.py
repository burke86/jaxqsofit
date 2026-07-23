"""Compatibility alias for shared spectral reparameterizations."""

import sys

from jaxsedfit import spectral_reparameterization as _impl

sys.modules[__name__] = _impl
