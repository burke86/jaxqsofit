"""Compatibility alias for spectral plotting now owned by jaxsedfit."""

import sys

from jaxsedfit import spectral_plotting as _impl

sys.modules[__name__] = _impl
