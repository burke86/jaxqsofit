from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

from .custom_components import CustomComponentSpec, make_custom_component
from .model import negative_gaussian_bal_component

MINSCA_DEFAULT = 0.0
MAXSCA_DEFAULT = 1e10

inisig_broad = 5e-3
minsig_broad = 0.004
maxsig_broad = 0.05

inisig_narrow = 1e-3
minsig_narrow = 2.3e-4
maxsig_narrow = 0.00169

inisig_narrow_relaxed = 1e-3
minsig_narrow_relaxed = 5e-4
maxsig_narrow_relaxed = maxsig_narrow

inisig_narrow_uv = 1e-3
minsig_narrow_uv = 3.333e-4
maxsig_narrow_uv = maxsig_narrow

inisig_oiii_wing = 3e-3
minsig_oiii_wing = minsig_narrow
maxsig_oiii_wing = 0.004

inisig_uv_broad = 5e-3
minsig_uv_broad = 0.002
maxsig_uv_broad = 0.05

inisig_nv = 2e-3
minsig_nv = 0.001
maxsig_nv = 0.01

voff_broad = 0.015
voff_broad_balmer = 0.01
voff_narrow = 0.01
voff_narrow_tight = 5e-3
voff_uv_broad = 0.015
voff_lya = 0.02
voff_nv = 0.005
voff_elg = 0.01
voff_elg_red = 0.008


def _line_row(
    *,
    lam: float,
    compname: str,
    minwav: float,
    maxwav: float,
    linename: str,
    ngauss: int = 1,
    inisca: float = 0.0,
    minsca: float = MINSCA_DEFAULT,
    maxsca: float = MAXSCA_DEFAULT,
    inisig: float,
    minsig: float,
    maxsig: float,
    voff: float,
    vindex: int,
    windex: int,
    findex: int,
    fvalue: float,
    vary: int = 1,
) -> Dict[str, Any]:
    return {
        "lambda": lam,
        "compname": compname,
        "minwav": minwav,
        "maxwav": maxwav,
        "linename": linename,
        "ngauss": ngauss,
        "inisca": inisca,
        "minsca": minsca,
        "maxsca": maxsca,
        "inisig": inisig,
        "minsig": minsig,
        "maxsig": maxsig,
        "voff": voff,
        "vindex": vindex,
        "windex": windex,
        "findex": findex,
        "fvalue": fvalue,
        "vary": vary,
    }


# Default line table in plain dict rows (same schema as notebook line config).
DEFAULT_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    # Halpha complex
    _line_row(lam=6564.61, compname='Ha', minwav=6400, maxwav=6800, linename='Ha_br', ngauss=2, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=6564.61, compname='Ha', minwav=6400, maxwav=6800, linename='Ha_na', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=maxsig_narrow_relaxed, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=6549.85, compname='Ha', minwav=6400, maxwav=6800, linename='NII6549', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=1, fvalue=0.001),
    _line_row(lam=6585.28, compname='Ha', minwav=6400, maxwav=6800, linename='NII6585', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=1, fvalue=0.003),
    _line_row(lam=6718.29, compname='Ha', minwav=6400, maxwav=6800, linename='SII6718', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=2, fvalue=0.001),
    _line_row(lam=6732.67, compname='Ha', minwav=6400, maxwav=6800, linename='SII6732', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=2, fvalue=0.001),
    # Hbeta / [OIII]
    _line_row(lam=4862.68, compname='Hb', minwav=4640, maxwav=5100, linename='Hb_br', ngauss=2, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=4862.68, compname='Hb', minwav=4640, maxwav=5100, linename='Hb_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=4960.30, compname='Hb', minwav=4640, maxwav=5100, linename='OIII4959c', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=5008.24, compname='Hb', minwav=4640, maxwav=5100, linename='OIII5007c', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.004),
    _line_row(lam=4960.30, compname='Hb', minwav=4640, maxwav=5100, linename='OIII4959w', inisig=inisig_oiii_wing, minsig=minsig_oiii_wing, maxsig=maxsig_oiii_wing, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    _line_row(lam=5008.24, compname='Hb', minwav=4640, maxwav=5100, linename='OIII5007w', inisig=inisig_oiii_wing, minsig=minsig_oiii_wing, maxsig=maxsig_oiii_wing, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.002),
    # Higher-order Balmer
    _line_row(lam=4341.68, compname='Hg', minwav=4200, maxwav=4400, linename='Hg_br', inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=4341.68, compname='Hg', minwav=4200, maxwav=4400, linename='Hg_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=4102.89, compname='Hd', minwav=4000, maxwav=4150, linename='Hd_br', inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=4102.89, compname='Hd', minwav=4000, maxwav=4150, linename='Hd_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    # Other optical/UV
    # CaII3934
    _line_row(lam=3728.48, compname='OII', minwav=3650, maxwav=3800, linename='OII3728', inisig=inisig_narrow_uv, minsig=minsig_narrow_uv, maxsig=maxsig_narrow_uv, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=3426.84, compname='NeV', minwav=3380, maxwav=3480, linename='NeV3426', inisig=inisig_narrow_uv, minsig=minsig_narrow_uv, maxsig=maxsig_narrow_uv, voff=voff_narrow, vindex=0, windex=0, findex=0, fvalue=0.001),
    # Mg II complex
    _line_row(lam=2798.75, compname='MgII', minwav=2700, maxwav=2900, linename='MgII_br', ngauss=2, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=2798.75, compname='MgII', minwav=2700, maxwav=2900, linename='MgII_na', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=maxsig_narrow_relaxed, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    # CIII complex
    _line_row(lam=1892.03, compname='CIII', minwav=1700, maxwav=1970, linename='SiIII1892', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=0.003, vindex=1, windex=1, findex=0, fvalue=0.005),
    _line_row(lam=1857.40, compname='CIII', minwav=1700, maxwav=1970, linename='AlIII1857', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=0.003, vindex=1, windex=1, findex=0, fvalue=0.005),
    _line_row(lam=1816.98, compname='CIII', minwav=1700, maxwav=1970, linename='SiII1816', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.0002),
    _line_row(lam=1750.26, compname='CIII', minwav=1700, maxwav=1970, linename='NIII1750', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    _line_row(lam=1718.55, compname='CIII', minwav=1700, maxwav=1900, linename='NIV1718', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    # CIV complex
    _line_row(lam=1549.06, compname='CIV', minwav=1500, maxwav=1700, linename='CIV_br', ngauss=3, inisig=inisig_uv_broad, minsig=0.001, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=1640.42, compname='CIV', minwav=1500, maxwav=1700, linename='HeII1640', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=0.002, voff=voff_elg_red, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=1663.48, compname='CIV', minwav=1500, maxwav=1700, linename='OIII1663', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=0.002, voff=voff_elg_red, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=1640.42, compname='CIV', minwav=1500, maxwav=1700, linename='HeII1640_br', inisig=inisig_uv_broad, minsig=0.0025, maxsig=0.02, voff=voff_elg_red, vindex=2, windex=2, findex=0, fvalue=0.002),
    _line_row(lam=1663.48, compname='CIV', minwav=1500, maxwav=1700, linename='OIII1663_br', inisig=inisig_uv_broad, minsig=0.0025, maxsig=0.02, voff=voff_elg_red, vindex=2, windex=2, findex=0, fvalue=0.002),
    # SiIV complex
    _line_row(lam=1402.06, compname='SiIV', minwav=1290, maxwav=1450, linename='SiIV_OIV1', inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=1, windex=1, findex=0, fvalue=0.05),
    _line_row(lam=1396.76, compname='SiIV', minwav=1290, maxwav=1450, linename='SiIV_OIV2', inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=1, windex=1, findex=0, fvalue=0.05),
    _line_row(lam=1335.30, compname='SiIV', minwav=1290, maxwav=1450, linename='CII1335', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    _line_row(lam=1304.35, compname='SiIV', minwav=1290, maxwav=1450, linename='OI1304', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    # Lya complex
    _line_row(lam=1215.67, compname='Lya', minwav=1150, maxwav=1290, linename='Lya_br', ngauss=3, inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_lya, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=1240.14, compname='Lya', minwav=1150, maxwav=1290, linename='NV1240', inisig=inisig_nv, minsig=minsig_nv, maxsig=maxsig_nv, voff=voff_nv, vindex=0, windex=0, findex=0, fvalue=0.002),
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
    _line_row(lam=3726.03, compname='OII', minwav=3650, maxwav=3800, linename='OII3726', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=31, fvalue=1.0),
    _line_row(lam=3728.82, compname='OII', minwav=3650, maxwav=3800, linename='OII3729', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=31, fvalue=1.0),
    _line_row(lam=3869.86, compname='NeIII', minwav=3800, maxwav=4020, linename='NeIII3869', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=3968.59, compname='NeIII', minwav=3900, maxwav=4100, linename='NeIII3968', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4102.89, compname='Hd', minwav=4000, maxwav=4150, linename='Hd_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4341.68, compname='Hg', minwav=4200, maxwav=4450, linename='Hg_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4364.44, compname='OIII', minwav=4300, maxwav=4450, linename='OIII4363', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4862.68, compname='Hb', minwav=4640, maxwav=5100, linename='Hb_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4687.02, compname='HeII', minwav=4620, maxwav=4760, linename='HeII4686', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4960.30, compname='OIII', minwav=4870, maxwav=5050, linename='OIII4959', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=32, fvalue=1.0),
    _line_row(lam=5008.24, compname='OIII', minwav=4920, maxwav=5100, linename='OIII5007', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=32, fvalue=1.0),
    _line_row(lam=5877.25, compname='HeI', minwav=5800, maxwav=5950, linename='HeI5876', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=6302.05, compname='OI', minwav=6200, maxwav=6420, linename='OI6300', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=33, fvalue=1.0),
    _line_row(lam=6365.54, compname='OI', minwav=6280, maxwav=6460, linename='OI6363', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=33, fvalue=1.0),
    _line_row(lam=6549.85, compname='NII', minwav=6460, maxwav=6640, linename='NII6548', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=34, fvalue=1.0),
    _line_row(lam=6564.61, compname='Ha', minwav=6480, maxwav=6660, linename='Ha_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=6585.28, compname='NII', minwav=6500, maxwav=6680, linename='NII6583', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=34, fvalue=1.0),
    _line_row(lam=6718.29, compname='SII', minwav=6640, maxwav=6800, linename='SII6716', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=35, fvalue=1.0),
    _line_row(lam=6732.67, compname='SII', minwav=6660, maxwav=6820, linename='SII6731', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=35, fvalue=1.0),
    # Red optical / far-red forbidden + He I
    _line_row(lam=7067.17, compname='HeI', minwav=7000, maxwav=7125, linename='HeI7065', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=7137.77, compname='ArIII', minwav=7050, maxwav=7220, linename='ArIII7138', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=7322.19, compname='OII', minwav=7260, maxwav=7375, linename='OII7320', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=22, fvalue=0.001),
    _line_row(lam=7332.97, compname='OII', minwav=7270, maxwav=7385, linename='OII7330', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=22, fvalue=0.001),
    _line_row(lam=7753.19, compname='ArIII', minwav=7680, maxwav=7820, linename='ArIII7751', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    # Paschen series (vacuum wavelengths, narrow by default)
    _line_row(lam=8752.87, compname='Paschen', minwav=8690, maxwav=8815, linename='Pa12', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=8865.22, compname='Paschen', minwav=8800, maxwav=8930, linename='Pa11', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=9017.38, compname='Paschen', minwav=8950, maxwav=9085, linename='Pa10', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=9231.55, compname='Paschen', minwav=9160, maxwav=9300, linename='Pa9', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=9548.59, compname='Paschen', minwav=9480, maxwav=9620, linename='Pae', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=10052.13, compname='Paschen', minwav=9980, maxwav=10130, linename='Pad', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=10941.09, compname='Paschen', minwav=10850, maxwav=11040, linename='Pag', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=12821.67, compname='Paschen', minwav=12700, maxwav=12950, linename='Pab', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=18756.13, compname='Paschen', minwav=18600, maxwav=18920, linename='Paa', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    # Strong red/NIR forbidden lines
    _line_row(lam=9071.09, compname='SIII', minwav=9000, maxwav=9135, linename='SIII9069', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=23, fvalue=0.001),
    _line_row(lam=9533.20, compname='SIII', minwav=9460, maxwav=9605, linename='SIII9531', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=23, fvalue=0.0025),
]

# Optional high-ionization/coronal narrow-line set.
DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    _line_row(lam=3346.79, compname='NeV', minwav=3300, maxwav=3385, linename='NeV3346', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=12, windex=12, findex=41, fvalue=1.0),
    _line_row(lam=3426.84, compname='NeV', minwav=3380, maxwav=3480, linename='NeV3426_hi', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=12, windex=12, findex=41, fvalue=1.0),
    _line_row(lam=5721.0, compname='FeVII', minwav=5660, maxwav=5785, linename='FeVII5721', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=6087.0, compname='FeVII', minwav=6030, maxwav=6145, linename='FeVII6087', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=6374.0, compname='FeX', minwav=6320, maxwav=6430, linename='FeX6374', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=7065.0, compname='HeI', minwav=7000, maxwav=7125, linename='HeI7065', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
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


def build_default_bal_components(flux: np.ndarray) -> tuple[CustomComponentSpec, ...]:
    """Return built-in BAL custom components with flux-scaled depth priors."""
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0

    def _bal_component(
        name: str,
        depth_frac: float,
        center: float,
        scale: float,
        low: float,
        high: float,
        sigma: float,
        sigma_scale: float = 0.35,
    ):
        return make_custom_component(
            name=name,
            parameter_priors={
                # Keep a zero-centered shrinkage prior, but with a broader scale so
                # moderate-to-deep troughs are still available when the data support them.
                "depth": {"dist": "HalfNormal", "scale": max(8.0 * depth_frac * fscale, 1e-6)},
                "center": {
                    # Force BAL trough centers to remain on the blue side of the
                    # corresponding broad emission line.
                    "dist": "TruncatedNormal",
                    "loc": float(center),
                    "scale": float(scale),
                    "low": float(low),
                    "high": float(high),
                },
                "sigma": {"dist": "LogNormal", "loc": np.log(float(sigma)), "scale": float(sigma_scale)},
                "shape_power": {
                    "dist": "TruncatedNormal",
                    "loc": 2.0,
                    "scale": 1.5,
                    "low": 2.0,
                    "high": 12.0,
                },
            },
            evaluate=negative_gaussian_bal_component,
        )

    return (
        _bal_component("bal_nv", depth_frac=0.04, center=1200.0, scale=70.0, low=1120.0, high=1240.0, sigma=22.0),
        # _bal_component("bal_nv_2", depth_frac=0.025, center=1160.0, scale=90.0, low=1100.0, high=1240.0, sigma=40.0),
        _bal_component("bal_siiv", depth_frac=0.04, center=1350.0, scale=70.0, low=1280.0, high=1397.0, sigma=22.0),
        # _bal_component("bal_siiv_2", depth_frac=0.025, center=1320.0, scale=90.0, low=1260.0, high=1397.0, sigma=40.0),
        _bal_component("bal_civ", depth_frac=0.05, center=1500.0, scale=80.0, low=1400.0, high=1549.0, sigma=24.0),
        # _bal_component("bal_civ_2", depth_frac=0.03, center=1450.0, scale=100.0, low=1350.0, high=1549.0, sigma=45.0),
        _bal_component("bal_ciii", depth_frac=0.03, center=1850.0, scale=80.0, low=1750.0, high=1909.0, sigma=30.0),
        # _bal_component("bal_ciii_2", depth_frac=0.02, center=1800.0, scale=100.0, low=1700.0, high=1909.0, sigma=50.0),
        _bal_component("bal_mgii", depth_frac=0.03, center=2798.0, scale=120.0, low=2750.0, high=2798.0, sigma=40.0),
        # _bal_component("bal_mgii_2", depth_frac=0.02, center=2760.0, scale=120.0, low=2700.0, high=2798.0, sigma=55.0),
    )


def build_default_prior_config(
    flux: np.ndarray,
    line_config: Dict[str, Any] | None = None,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
    pl_pivot: float | None = None,
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
    pl_pivot : float or None, optional
        Optional manual override for the power-law continuum pivot wavelength in
        Angstrom. If ``None``, the model uses the midpoint of the fitted rest-frame
        wavelength coverage.
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
        "log_cont_norm": {"dist": "LogNormal", "loc": np.log(max(fscale, 1e-8)), "scale": 0.3},
        "PL_norm": {"dist": "HalfNormal", "scale": max(0.5 * fscale, 1e-10)},
        "PL_slope": {"dist": "Normal", "loc": -1.5, "scale": 0.4},
        "PL_pivot": None if pl_pivot is None else float(pl_pivot),
        "reddening_ebv": {"dist": "HalfNormal", "scale": 0.3},
        "reddening_uv_ref": 2500.0,
        "reddening_alpha": 1.2,
        "log_frac_host": {"dist": "StudentT", "loc": 0.0, "scale": 2.0, "df": 3.0},
        "host_redshift_prior": {
            "enabled": False,
            "z_mid": 1.0,
            "width": 0.2,
            "lowz_loc_offset": 0.0,
            "highz_loc_offset": -8.0,
            "lowz_scale_mult": 1.0,
            "highz_scale_mult": 0.05,
            "lowz_df": 3.0,
            "highz_df": 20.0,
        },
        "tau_host": {"dist": "HalfNormal", "scale": 1.0},
        "raw_w": {"dist": "Normal", "loc": -0.5, "scale": 1.0},
        "gal_v_kms": {"dist": "Normal", "loc": 0.0, "scale": 120.0},
        "gal_sigma_kms": {"dist": "HalfNormal", "scale": 200.0},
        "log_Fe_uv_norm": {"dist": "LogNormal", "loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Fe_op_over_uv": {"dist": "Normal", "loc": 0.0, "scale": 0.05},
        "log_Fe_uv_FWHM": {"dist": "LogNormal", "loc": np.log(3000.0), "scale": 0.3},
        "log_Fe_op_FWHM": {"dist": "LogNormal", "loc": np.log(3000.0), "scale": 0.3},
        "Fe_uv_shift": {"dist": "Normal", "loc": 0.0, "scale": 1e-3},
        "Fe_op_shift": {"dist": "Normal", "loc": 0.0, "scale": 1e-3},
        "log_Balmer_norm": {"dist": "LogNormal", "loc": np.log(max(1e-3 * fscale, 1e-10)), "scale": 0.5},
        "log_Balmer_Tau": {"dist": "LogNormal", "loc": np.log(0.5), "scale": 0.25},
        "log_Balmer_vel": {"dist": "LogNormal", "loc": np.log(3000.0), "scale": 0.25},
        "poly_c1": {"dist": "Normal", "loc": 0.0, "scale": 0.1},
        "poly_c2": {"dist": "Normal", "loc": 0.0, "scale": 0.1},
        "poly_c3": {"dist": "Normal", "loc": 0.0, "scale": 0.05},
        "poly_c4": {"dist": "Normal", "loc": 0.0, "scale": 0.05},
        "poly_c5": {"dist": "Normal", "loc": 0.0, "scale": 0.03},
        "poly_c6": {"dist": "Normal", "loc": 0.0, "scale": 0.03},
        "frac_jitter": {"dist": "HalfNormal", "scale": 0.02},
        "add_jitter": {"dist": "HalfNormal", "scale_mult_err": 0.3},
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
