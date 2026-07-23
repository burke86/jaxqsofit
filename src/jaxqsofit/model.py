"""Compatibility alias for the spectral model now owned by jaxsedfit."""

import sys

from jaxsedfit import spectral_model as _impl

sys.modules[__name__] = _impl
