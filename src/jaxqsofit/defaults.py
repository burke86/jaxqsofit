from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

# Default line table in plain dict rows (same schema as notebook line config).
DEFAULT_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    # Halpha complex
    {'lambda': 6564.61, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'Ha_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 6564.61, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'Ha_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 5e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 6549.85, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'NII6549', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 6585.28, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'NII6585', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 0.003, 'vary': 1},
    {'lambda': 6718.29, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'SII6718', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 2, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 6732.67, 'compname': 'Ha', 'minwav': 6400, 'maxwav': 6800, 'linename': 'SII6732', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 5e-3, 'vindex': 1, 'windex': 1, 'findex': 2, 'fvalue': 0.001, 'vary': 1},
    # Hbeta / [OIII]
    {'lambda': 4862.68, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'Hb_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 4862.68, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'Hb_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 4960.30, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII4959c', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 5008.24, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII5007c', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.004, 'vary': 1},
    {'lambda': 4960.30, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII4959w', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 3e-3, 'minsig': 2.3e-4, 'maxsig': 0.004, 'voff': 0.01, 'vindex': 2, 'windex': 2, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 5008.24, 'compname': 'Hb', 'minwav': 4640, 'maxwav': 5100, 'linename': 'OIII5007w', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 3e-3, 'minsig': 2.3e-4, 'maxsig': 0.004, 'voff': 0.01, 'vindex': 2, 'windex': 2, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    # Higher-order Balmer
    {'lambda': 4341.68, 'compname': 'Hg', 'minwav': 4200, 'maxwav': 4400, 'linename': 'Hg_br', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 4341.68, 'compname': 'Hg', 'minwav': 4200, 'maxwav': 4400, 'linename': 'Hg_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 4102.89, 'compname': 'Hd', 'minwav': 4000, 'maxwav': 4150, 'linename': 'Hd_br', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 4102.89, 'compname': 'Hd', 'minwav': 4000, 'maxwav': 4150, 'linename': 'Hd_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    # Other optical/UV
    {'lambda': 3728.48, 'compname': 'OII', 'minwav': 3650, 'maxwav': 3800, 'linename': 'OII3728', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 3.333e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 3426.84, 'compname': 'NeV', 'minwav': 3380, 'maxwav': 3480, 'linename': 'NeV3426', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 3.333e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 2798.75, 'compname': 'MgII', 'minwav': 2700, 'maxwav': 2900, 'linename': 'MgII_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 2798.75, 'compname': 'MgII', 'minwav': 2700, 'maxwav': 2900, 'linename': 'MgII_na', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 5e-4, 'maxsig': 0.00169, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
    {'lambda': 1908.73, 'compname': 'CIII', 'minwav': 1700, 'maxwav': 1970, 'linename': 'CIII_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 99, 'windex': 0, 'findex': 0, 'fvalue': 0.01, 'vary': 1},
    {'lambda': 1549.06, 'compname': 'CIV', 'minwav': 1500, 'maxwav': 1700, 'linename': 'CIV_br', 'ngauss': 2, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.004, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 1402.06, 'compname': 'SiIV', 'minwav': 1290, 'maxwav': 1450, 'linename': 'SiIV_OIV1', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.002, 'maxsig': 0.05, 'voff': 0.015, 'vindex': 1, 'windex': 1, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 1215.67, 'compname': 'Lya', 'minwav': 1150, 'maxwav': 1290, 'linename': 'Lya_br', 'ngauss': 3, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 5e-3, 'minsig': 0.002, 'maxsig': 0.05, 'voff': 0.02, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.05, 'vary': 1},
    {'lambda': 1240.14, 'compname': 'Lya', 'minwav': 1150, 'maxwav': 1290, 'linename': 'NV1240', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 2e-3, 'minsig': 0.001, 'maxsig': 0.01, 'voff': 0.005, 'vindex': 0, 'windex': 0, 'findex': 0, 'fvalue': 0.002, 'vary': 1},
]

DEFAULT_LINE_CONFIG: Dict[str, Any] = {
    "line_dmu_scale_mult": 0.25,
    "line_sig_scale_mult": 0.25,
    "line_amp_scale_mult": 0.25,
    "line": {"table": DEFAULT_LINE_PRIOR_ROWS},
}

# Additional narrow lines commonly used for emission-line galaxies (ELGs).
# These can be appended to the default line list via
# build_default_prior_config(..., include_elg_narrow_lines=True).
DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    {'lambda': 3726.03, 'compname': 'OII',   'minwav': 3650, 'maxwav': 3800, 'linename': 'OII3726',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 31, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 3728.82, 'compname': 'OII',   'minwav': 3650, 'maxwav': 3800, 'linename': 'OII3729',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 31, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 3869.86, 'compname': 'NeIII', 'minwav': 3800, 'maxwav': 4020, 'linename': 'NeIII3869',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 3968.59, 'compname': 'NeIII', 'minwav': 3900, 'maxwav': 4100, 'linename': 'NeIII3968',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 4102.89, 'compname': 'Hd',    'minwav': 4000, 'maxwav': 4150, 'linename': 'Hd_na_elg',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 4341.68, 'compname': 'Hg',    'minwav': 4200, 'maxwav': 4450, 'linename': 'Hg_na_elg',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 4364.44, 'compname': 'OIII',  'minwav': 4300, 'maxwav': 4450, 'linename': 'OIII4363',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 4862.68, 'compname': 'Hb',    'minwav': 4640, 'maxwav': 5100, 'linename': 'Hb_na_elg',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 4687.02, 'compname': 'HeII',  'minwav': 4620, 'maxwav': 4760, 'linename': 'HeII4686',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 4960.30, 'compname': 'OIII',  'minwav': 4870, 'maxwav': 5050, 'linename': 'OIII4959',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 32, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 5008.24, 'compname': 'OIII',  'minwav': 4920, 'maxwav': 5100, 'linename': 'OIII5007',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 32, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 5877.25, 'compname': 'HeI',   'minwav': 5800, 'maxwav': 5950, 'linename': 'HeI5876',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 6302.05, 'compname': 'OI',    'minwav': 6200, 'maxwav': 6420, 'linename': 'OI6300',     'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 33, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 6365.54, 'compname': 'OI',    'minwav': 6280, 'maxwav': 6460, 'linename': 'OI6363',     'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 33, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 6549.85, 'compname': 'NII',   'minwav': 6460, 'maxwav': 6640, 'linename': 'NII6548',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 34, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 6564.61, 'compname': 'Ha',    'minwav': 6480, 'maxwav': 6660, 'linename': 'Ha_na_elg',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 6585.28, 'compname': 'NII',   'minwav': 6500, 'maxwav': 6680, 'linename': 'NII6583',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 34, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 6718.29, 'compname': 'SII',   'minwav': 6640, 'maxwav': 6800, 'linename': 'SII6716',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 35, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 6732.67, 'compname': 'SII',   'minwav': 6660, 'maxwav': 6820, 'linename': 'SII6731',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 35, 'fvalue': 1.0,  'vary': 1},
    # Red optical / far-red forbidden + He I
    {'lambda': 7067.17, 'compname': 'HeI',   'minwav': 7000, 'maxwav': 7125, 'linename': 'HeI7065',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 7137.77, 'compname': 'ArIII', 'minwav': 7050, 'maxwav': 7220, 'linename': 'ArIII7138', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 7322.19, 'compname': 'OII',   'minwav': 7260, 'maxwav': 7375, 'linename': 'OII7320',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 22, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 7332.97, 'compname': 'OII',   'minwav': 7270, 'maxwav': 7385, 'linename': 'OII7330',   'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 22, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 7753.19, 'compname': 'ArIII', 'minwav': 7680, 'maxwav': 7820, 'linename': 'ArIII7751', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    # Paschen series (vacuum wavelengths, narrow by default)
    {'lambda': 8752.87, 'compname': 'Paschen', 'minwav': 8690, 'maxwav': 8815, 'linename': 'Pa12', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 8865.22, 'compname': 'Paschen', 'minwav': 8800, 'maxwav': 8930, 'linename': 'Pa11', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 9017.38, 'compname': 'Paschen', 'minwav': 8950, 'maxwav': 9085, 'linename': 'Pa10', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 9231.55, 'compname': 'Paschen', 'minwav': 9160, 'maxwav': 9300, 'linename': 'Pa9',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 9548.59, 'compname': 'Paschen', 'minwav': 9480, 'maxwav': 9620, 'linename': 'Pae',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 10052.13,'compname': 'Paschen', 'minwav': 9980, 'maxwav': 10130,'linename': 'Pad',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 10941.09,'compname': 'Paschen', 'minwav': 10850,'maxwav': 11040,'linename': 'Pag',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 12821.67,'compname': 'Paschen', 'minwav': 12700,'maxwav': 12950,'linename': 'Pab',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 18756.13,'compname': 'Paschen', 'minwav': 18600,'maxwav': 18920,'linename': 'Paa',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0, 'fvalue': 0.001, 'vary': 1},
    # Strong red/NIR forbidden lines
    {'lambda': 9071.09, 'compname': 'SIII', 'minwav': 9000, 'maxwav': 9135, 'linename': 'SIII9069', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 23, 'fvalue': 0.001, 'vary': 1},
    {'lambda': 9533.20, 'compname': 'SIII', 'minwav': 9460, 'maxwav': 9605, 'linename': 'SIII9531', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 11, 'windex': 11, 'findex': 23, 'fvalue': 0.0025, 'vary': 1},
]

# Optional high-ionization/coronal narrow-line set.
DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    {'lambda': 3346.79, 'compname': 'NeV',   'minwav': 3300, 'maxwav': 3385, 'linename': 'NeV3346',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 12, 'windex': 12, 'findex': 41, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 3426.84, 'compname': 'NeV',   'minwav': 3380, 'maxwav': 3480, 'linename': 'NeV3426_hi', 'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.01,  'vindex': 12, 'windex': 12, 'findex': 41, 'fvalue': 1.0,  'vary': 1},
    {'lambda': 5721.0,  'compname': 'FeVII', 'minwav': 5660, 'maxwav': 5785, 'linename': 'FeVII5721',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 6087.0,  'compname': 'FeVII', 'minwav': 6030, 'maxwav': 6145, 'linename': 'FeVII6087',  'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 6374.0,  'compname': 'FeX',   'minwav': 6320, 'maxwav': 6430, 'linename': 'FeX6374',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
    {'lambda': 7065.0,  'compname': 'HeI',   'minwav': 7000, 'maxwav': 7125, 'linename': 'HeI7065',    'ngauss': 1, 'inisca': 0.0, 'minsca': 0.0, 'maxsca': 1e10, 'inisig': 1e-3, 'minsig': 2.3e-4, 'maxsig': 0.00169, 'voff': 0.008, 'vindex': 12, 'windex': 12, 'findex': 0,  'fvalue': 0.001, 'vary': 1},
]


def _apply_robust_line_scale_priors(
    line_rows: List[Dict[str, Any]],
    fscale: float,
    fmax: float,
) -> List[Dict[str, Any]]:
    """Apply flux-aware robust bounds/initialization to line-scale priors."""
    if len(line_rows) == 0:
        return line_rows

    # Keep dynamic range positive even for nearly flat/noisy spectra.
    delta = max(float(fmax - fscale), 0.1 * float(fscale), 1e-8)

    for row in line_rows:
        linename = str(row.get("linename", "")).lower()
        is_broad = linename.endswith("_br") or ("_br" in linename)

        maxsca = float(row.get("maxsca", np.inf))
        minsca = float(row.get("minsca", 0.0))
        inisca = float(row.get("inisca", 0.0))

        # Broad lines get a tighter cap than narrow lines by default.
        if is_broad:
            max_cap = 1.0 * delta
        else:
            max_cap = 1.2 * delta
        maxsca = min(maxsca, max_cap)

        # Keep scales strictly positive and ordered.
        mins_floor = max(minsca, 1e-4 * float(fscale), 1e-12)
        maxsca = max(maxsca, 1.01 * mins_floor)
        inisca = float(np.clip(inisca, mins_floor, maxsca))

        row["minsca"] = mins_floor
        row["maxsca"] = maxsca
        row["inisca"] = inisca

    return line_rows


def _append_unique_by_wavelength(
    base_rows: List[Dict[str, Any]],
    extra_rows: List[Dict[str, Any]],
    atol_angstrom: float = 1.0,
) -> List[Dict[str, Any]]:
    """Append rows from `extra_rows` only if no near-duplicate wavelength exists."""
    out = list(base_rows)
    for row in extra_rows:
        lam_new = float(row.get("lambda", np.nan))
        if not np.isfinite(lam_new):
            continue
        exists = False
        for old in out:
            lam_old = float(old.get("lambda", np.nan))
            if np.isfinite(lam_old) and abs(lam_old - lam_new) <= float(atol_angstrom):
                exists = True
                break
        if not exists:
            out.append(row)
    return out


def build_default_prior_config(
    flux: np.ndarray,
    line_config: Dict[str, Any] | None = None,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
) -> Dict[str, Any]:
    """Build a full prior_config with sane defaults from data flux scale.

    Parameters
    ----------
    flux : ndarray
        Input flux array used to set data-scale-aware defaults.
    line_config : dict or None, optional
        Optional line configuration override. If None, default line config is used.
    include_elg_narrow_lines : bool, optional
        If True, append additional narrow ELG lines from
        ``DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS`` to the active line table.
    include_high_ionization_lines : bool, optional
        If True, append additional high-ionization lines from
        ``DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS`` to the active line table.
    """
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    fmax = float(np.nanmax(np.abs(f[finite]))) if np.any(finite) else fscale
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = fscale

    cfg: Dict[str, Any] = {
        "log_cont_norm": {"loc": np.log(max(fscale, 1e-8)), "scale": 0.3},
        "PL_slope": {"loc": -1.5, "scale": 0.4, "low": -3.5, "high": 0.3},
        "log_frac_host": {"loc": 0.0, "scale": 2.0, "df": 3.0},
        "tau_host": {"scale": 1.0},
        "raw_w": {"loc": -0.5, "scale": 1.0},
        "gal_v_kms": {"loc": 0.0, "scale": 120.0},
        "gal_sigma_kms": {"scale": 200.0},
        "log_Fe_uv_norm": {"loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Fe_op_over_uv": {"loc": 0.0, "scale": 0.05},
        "log_Fe_uv_FWHM": {"loc": np.log(3000.0), "scale": 0.3},
        "log_Fe_op_FWHM": {"loc": np.log(3000.0), "scale": 0.3},
        "Fe_uv_shift": {"loc": 0.0, "scale": 1e-3},
        "Fe_op_shift": {"loc": 0.0, "scale": 1e-3},
        "log_Balmer_norm": {"loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Balmer_Tau": {"loc": np.log(0.5), "scale": 0.25},
        "log_Balmer_vel": {"loc": np.log(3000.0), "scale": 0.25},
        "poly_c1": {"loc": 0.0, "scale": 0.1},
        "poly_c2": {"loc": 0.0, "scale": 0.1},
        "poly_c3": {"loc": 0.0, "scale": 0.05},
        "poly_c4": {"loc": 0.0, "scale": 0.05},
        "poly_c5": {"loc": 0.0, "scale": 0.03},
        "poly_c6": {"loc": 0.0, "scale": 0.03},
        "edge_rbf_amp": {"loc": 0.0, "scale": 0.05},
        "log_edge_rbf_sigma_blue": {"loc": np.log(80.0), "scale": 0.4},
        "log_edge_rbf_sigma_red": {"loc": np.log(80.0), "scale": 0.4},
        "edge_rbf_n_per_side": 3,
        "edge_rbf_frac_min": 0.01,
        "edge_rbf_frac_max": 0.10,
        "frac_jitter": {"scale": 0.02},
        "add_jitter": {"scale_mult_err": 0.3},
        "student_t_df": 3.0,
        "out_params": {
            "cont_loc": [1350.0, 2500.0, 3000.0, 4200.0, 5100.0],
        },
    }

    lc = copy.deepcopy(DEFAULT_LINE_CONFIG if line_config is None else line_config)
    if isinstance(lc, dict):
        line_cfg = lc.get("line", {})
        if isinstance(line_cfg, dict):
            table = line_cfg.get("table", None)
            if isinstance(table, list):
                if include_elg_narrow_lines:
                    table = _append_unique_by_wavelength(
                        list(table),
                        copy.deepcopy(DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS),
                        atol_angstrom=1.0,
                    )
                if include_high_ionization_lines:
                    table = _append_unique_by_wavelength(
                        list(table),
                        copy.deepcopy(DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS),
                        atol_angstrom=1.0,
                    )
                line_cfg["table"] = _apply_robust_line_scale_priors(table, fscale=fscale, fmax=fmax)
    cfg.update(lc)
    return cfg
