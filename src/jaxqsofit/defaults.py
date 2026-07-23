"""Compatibility alias for spectral defaults now owned by jaxsedfit."""

import sys

from jaxsedfit import spectral_defaults as _impl

sys.modules[__name__] = _impl
