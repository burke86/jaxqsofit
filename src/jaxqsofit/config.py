"""Compatibility alias for configuration now owned by :mod:`jaxsedfit`."""

import sys

from jaxsedfit import spectral_config as _impl

sys.modules[__name__] = _impl
