from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import numpyro.distributions as dist

from .config import PriorConfig
from .custom_components import CustomComponentSpec, make_custom_component
from .model import gaussian_bal_optical_depth_component

MINSCA_DEFAULT = 0.0
MAXSCA_DEFAULT = 1e10
AMPLITUDE_FLOOR = 1e-32
ROBUST_FLUX_HIGH_PERCENTILE = 99.5

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
) -> PriorConfig:
    """Build one line-prior row.

    Wavelength fields are rest-frame vacuum Angstroms, matching SDSS spectra
    and the rest-frame wavelength grid used by the fitter.

    Parameters
    ----------
    lam : object
        lam value.
    compname : object
        compname value.
    linename : object
        linename value.
    ngauss : object
        ngauss value.
    inisca : object
        inisca value.
    minsca : object
        minsca value.
    maxsca : object
        maxsca value.
    inisig : object
        inisig value.
    minsig : object
        minsig value.
    maxsig : object
        maxsig value.
    voff : object
        voff value.
    vindex : object
        vindex value.
    windex : object
        windex value.
    findex : object
        findex value.
    fvalue : object
        fvalue value.
    vary : object
        vary value.
    """
    return {
        "lambda": lam,
        "compname": compname,
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


def _lnlam_peak_ratio_for_flux_ratio(
    flux_ratio: float,
    numerator_lam: float,
    denominator_lam: float,
) -> float:
    """Convert an integrated-flux ratio to a tied peak-amplitude ratio.

    Line ties are applied to Gaussian peak amplitudes in ln-lambda space. For
    equal ln-lambda widths, integrated flux scales as peak * rest wavelength.

    Parameters
    ----------
    flux_ratio : object
        flux_ratio value.
    numerator_lam : object
        numerator_lam value.
    denominator_lam : object
        denominator_lam value.
    """
    return flux_ratio * denominator_lam / numerator_lam


"""
Default line-prior table.

Each row below defines one emission-line prior in the same plain-dict schema
accepted by notebook line configs. The table is converted into NumPy/JAX
metadata by ``build_tied_line_meta_from_linelist`` before sampling.

Coordinate system
-----------------
``lambda`` is the rest-frame vacuum wavelength in Angstroms. The Gaussian
model itself is evaluated in ln(lambda), not linear wavelength. Consequently,
``inisig``, ``minsig``, and ``maxsig`` are Gaussian widths in ln(lambda). For
small widths, these are approximately velocity widths divided by c. For
example, ``sigma_ln_lambda = 0.001`` corresponds to roughly 300 km/s Gaussian
sigma, or about 700 km/s FWHM.

Amplitude and width fields
--------------------------
``inisca``, ``minsca``, and ``maxsca`` are priors on the Gaussian peak
amplitude. They are not integrated line fluxes. The integrated flux of a
single Gaussian in linear wavelength scales approximately as
peak_amplitude * sigma_ln_lambda * lambda0. This matters for fixed doublet
ratios: the helper ``_lnlam_peak_ratio_for_flux_ratio`` converts an intended
integrated-flux ratio into the peak-amplitude ratio required by the ln-lambda
Gaussian model, assuming the tied components share the same width.

``inisig``, ``minsig``, and ``maxsig`` define the prior for the Gaussian
ln-lambda width group. If multiple rows share a nonzero ``windex`` within the
same component complex, they share one sampled width. If ``windex`` is zero,
the row is not tied to other rows by width.

Velocity offsets: ``voff`` and ``vindex``
----------------------------------------
``voff`` is the allowed absolute center shift in ln(lambda): the sampled
center offset is constrained to ``[-voff, +voff]`` around ``log(lambda)``.
For small offsets this is approximately a velocity range of ``voff * c``.

``vindex`` controls tied velocity shifts. Rows with the same positive
``vindex`` within the same component complex share one sampled center offset.
Rows with ``vindex=0`` are independent. This follows the PyQSOFit convention
that only nonzero tie indices are constraints, but jaxqsofit additionally
scopes the tie by component complex to avoid accidental cross-complex tying
when the same integer is reused in different wavelength regions.

Width ties: ``windex``
---------------------
``windex`` works like ``vindex``, but for Gaussian width. Rows with the same
positive ``windex`` within the same component complex share one sampled
``sigma_ln_lambda``. Rows with ``windex=0`` are independent. This is commonly
used for physically related doublets or narrow-line complexes whose velocity
widths should match.

Amplitude/flux-ratio ties: ``findex`` and ``fvalue``
---------------------------------------------------
PyQSOFit documents this rule as: entries with the same nonzero ``findex`` have
constrained flux ratios. In this implementation, the same convention is used
with two precise details:

1. Only positive ``findex`` values tie rows together. ``findex=0`` means the
   row gets its own independent amplitude group.
2. Ties are local to the component complex, represented internally by
   ``compname``. The same positive ``findex`` can therefore be reused in
   different complexes without coupling unrelated lines such as Ha and Hb.

Within each tied amplitude group, one sampled peak-amplitude parameter is
created. Each component's peak amplitude is then
``line_amp_group[fgroup] * flux_ratio``, where ``flux_ratio`` is derived from
that row's ``fvalue`` relative to the first row in the group. Thus ``fvalue``
sets the fixed relative peak amplitude inside a tied group. For equal
ln-lambda widths, choosing ``fvalue`` with
``_lnlam_peak_ratio_for_flux_ratio`` enforces the desired integrated-flux
ratio.

For untied rows with ``findex=0``, ``fvalue`` is not a fixed flux ratio. It is
only the initial/default amplitude scale used to seed that independent
amplitude group's prior.

Narrow-line centroid pooling
----------------------------
By default, narrow cores are pooled into low-ionization, high-ionization, and
coronal kinematic families. All lines in a family share one exact centroid and
one exact FWHM across complexes. No complex-specific offsets or wavelength-
calibration error terms are added. Broad components and explicitly identified
wings or outflows are excluded. Set
``LineConfig.pool_narrow_centroids=False`` to restore the line-table centroid
and width ties without cross-complex family pooling.

Multiple Gaussians: ``ngauss``
-----------------------------
``ngauss`` expands one row into multiple Gaussian components with names like
``CIV_br_1``, ``CIV_br_2``, etc. Each expanded Gaussian is intentionally given
an independent internal tie label. For broad-line rows with ``ngauss > 1``,
their widths are sampled in strictly increasing order to remove equivalent
label-switched posterior modes. Their centroids use a shared broad-line shift
plus zero-sum relative offsets. Peak amplitudes remain independent. If a
genuinely tied multi-component structure is needed, write the components as
explicit rows with shared positive tie indices.

Line naming and plotting
------------------------
``linename`` is the output component name used in model metadata and plots.
The current plotting convention draws names containing ``"_br"`` and [O III]
wing names ending in ``"w"`` with broad-component styling; other built-in line
names use narrow-component styling.
``compname`` is used for grouping/tie scoping by line complex; it is not just a
display label.
"""
# Default line table in plain dict rows (same schema as notebook line config).
DEFAULT_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    # Halpha complex
    _line_row(lam=6564.61, compname='Ha', linename='Ha_br', ngauss=2, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=6564.61, compname='Ha', linename='Ha_na', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=maxsig_narrow_relaxed, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=6549.85, compname='Ha', linename='NII6549', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=1, fvalue=1.0),
    _line_row(lam=6585.28, compname='Ha', linename='NII6585', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=1, fvalue=_lnlam_peak_ratio_for_flux_ratio(3.0, 6585.28, 6549.85)),
    _line_row(lam=6718.29, compname='Ha', linename='SII6718', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=2, fvalue=0.001),
    _line_row(lam=6732.67, compname='Ha', linename='SII6732', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow_tight, vindex=1, windex=1, findex=2, fvalue=0.001),
    # Hbeta / [OIII]
    _line_row(lam=4862.68, compname='Hb', linename='Hb_br', ngauss=2, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=4862.68, compname='Hb', linename='Hb_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=4960.30, compname='Hb', linename='OIII4959c', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=3, fvalue=1.0),
    _line_row(lam=5008.24, compname='Hb', linename='OIII5007c', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=3, fvalue=_lnlam_peak_ratio_for_flux_ratio(2.98, 5008.24, 4960.30)),
    _line_row(lam=4960.30, compname='Hb', linename='OIII4959w', inisig=inisig_oiii_wing, minsig=minsig_oiii_wing, maxsig=maxsig_oiii_wing, voff=voff_narrow, vindex=2, windex=2, findex=4, fvalue=1.0),
    _line_row(lam=5008.24, compname='Hb', linename='OIII5007w', inisig=inisig_oiii_wing, minsig=minsig_oiii_wing, maxsig=maxsig_oiii_wing, voff=voff_narrow, vindex=2, windex=2, findex=4, fvalue=_lnlam_peak_ratio_for_flux_ratio(2.98, 5008.24, 4960.30)),
    # Higher-order Balmer
    _line_row(lam=4341.68, compname='Hg', linename='Hg_br', inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=4341.68, compname='Hg', linename='Hg_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=4102.89, compname='Hd', linename='Hd_br', inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=4102.89, compname='Hd', linename='Hd_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    # Other optical/UV
    _line_row(lam=3728.48, compname='OII', linename='OII3728', inisig=inisig_narrow_uv, minsig=minsig_narrow_uv, maxsig=maxsig_narrow_uv, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=3426.84, compname='NeV', linename='NeV3426', inisig=inisig_narrow_uv, minsig=minsig_narrow_uv, maxsig=maxsig_narrow_uv, voff=voff_narrow, vindex=0, windex=0, findex=0, fvalue=0.001),
    # Principal Paschen lines. Each row has findex=0 so the broad- and
    # narrow-line amplitudes, and the amplitudes of different transitions,
    # remain independent rather than imposing Case-B ratios on the AGN BLR.
    _line_row(lam=9548.59, compname='Pae', linename='Pae_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=10052.13, compname='Pad', linename='Pad_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=10941.09, compname='Pag', linename='Pag_br', ngauss=1, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=10941.09, compname='Pag', linename='Pag_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=10833.31, compname='HeI10830', linename='HeI10830_br', ngauss=1, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad, vindex=0, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=10833.31, compname='HeI10830', linename='HeI10830_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=12821.67, compname='Pab', linename='Pab_br', ngauss=1, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.02),
    _line_row(lam=12821.67, compname='Pab', linename='Pab_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    _line_row(lam=18756.13, compname='Paa', linename='Paa_br', ngauss=1, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad_balmer, vindex=0, windex=0, findex=0, fvalue=0.02),
    _line_row(lam=18756.13, compname='Paa', linename='Paa_na', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.001),
    # Mg II complex
    _line_row(lam=2798.75, compname='MgII', linename='MgII_br', ngauss=2, inisig=inisig_broad, minsig=minsig_broad, maxsig=maxsig_broad, voff=voff_broad, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=2798.75, compname='MgII', linename='MgII_na', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=maxsig_narrow_relaxed, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    # CIII complex
    _line_row(lam=1908.73, compname='CIII', linename='CIII_br', ngauss=2, inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=3, windex=0, findex=0, fvalue=0.01),
    _line_row(lam=1908.73, compname='CIII', linename='CIII_na', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=0.002, voff=voff_narrow, vindex=4, windex=4, findex=0, fvalue=0.002),
    _line_row(lam=1892.03, compname='CIII', linename='SiIII1892', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=0.003, vindex=1, windex=1, findex=0, fvalue=0.005),
    _line_row(lam=1857.40, compname='CIII', linename='AlIII1857', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=0.003, vindex=1, windex=1, findex=0, fvalue=0.005),
    _line_row(lam=1816.98, compname='CIII', linename='SiII1816', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.0002),
    _line_row(lam=1750.26, compname='CIII', linename='NIII1750', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    _line_row(lam=1718.55, compname='CIII', linename='NIV1718', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    # CIV complex
    _line_row(lam=1549.06, compname='CIV', linename='CIV_br', ngauss=3, inisig=inisig_uv_broad, minsig=0.001, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=1549.06, compname='CIV', linename='CIV_na', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=0.002, voff=voff_narrow, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=1663.48, compname='CIV', linename='OIII1663', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=0.002, voff=voff_elg_red, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=1663.48, compname='CIV', linename='OIII1663_br', inisig=inisig_uv_broad, minsig=0.0025, maxsig=0.02, voff=voff_elg_red, vindex=2, windex=2, findex=0, fvalue=0.002),
    _line_row(lam=1640.42, compname='CIV', linename='HeII1640', inisig=inisig_narrow_relaxed, minsig=minsig_narrow_relaxed, maxsig=0.002, voff=voff_elg_red, vindex=1, windex=1, findex=0, fvalue=0.002),
    _line_row(lam=1640.42, compname='CIV', linename='HeII1640_br', inisig=inisig_uv_broad, minsig=0.0025, maxsig=0.02, voff=voff_elg_red, vindex=2, windex=2, findex=0, fvalue=0.002),
    # SiIV complex
    _line_row(lam=1402.06, compname='SiIV', linename='SiIV_OIV1_br', inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=1, windex=1, findex=0, fvalue=0.05),
    _line_row(lam=1396.76, compname='SiIV', linename='SiIV_OIV2_br', inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_uv_broad, vindex=1, windex=1, findex=0, fvalue=0.05),
    _line_row(lam=1335.30, compname='SiIV', linename='CII1335', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    _line_row(lam=1304.35, compname='SiIV', linename='OI1304', inisig=inisig_nv, minsig=minsig_nv, maxsig=0.015, voff=voff_narrow, vindex=2, windex=2, findex=0, fvalue=0.001),
    # Lya complex
    _line_row(lam=1215.67, compname='Lya', linename='Lya_br', ngauss=3, inisig=inisig_uv_broad, minsig=minsig_uv_broad, maxsig=maxsig_uv_broad, voff=voff_lya, vindex=0, windex=0, findex=0, fvalue=0.05),
    _line_row(lam=1240.14, compname='Lya', linename='NV1240_br', inisig=inisig_nv, minsig=minsig_nv, maxsig=maxsig_nv, voff=voff_nv, vindex=0, windex=0, findex=0, fvalue=0.002),
]

DEFAULT_LINE_CONFIG: Dict[str, Any] = {
    "line_dmu_scale_mult": 0.25,
    "line_sig_scale_mult": 0.25,
    "line_amp_scale_mult": 0.25,
    # Suppress unsupported second-and-later Gaussians while allowing the data
    # to retain them when a broad-line profile genuinely needs extra structure.
    "line_extra_amp_scale_mult": 0.5,
    "line": {"table": DEFAULT_LINE_PRIOR_ROWS},
}

# Additional narrow lines commonly used for emission-line galaxies (ELGs).
# These can be appended to the default line list via
# _build_default_prior_config(..., include_elg_narrow_lines=True).
DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    _line_row(lam=3726.03, compname='OII', linename='OII3726', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=31, fvalue=1.0),
    _line_row(lam=3728.82, compname='OII', linename='OII3729', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=31, fvalue=1.0),
    _line_row(lam=3869.86, compname='NeIII', linename='NeIII3869', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=3968.59, compname='NeIII', linename='NeIII3968', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4102.89, compname='Hd', linename='Hd_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4341.68, compname='Hg', linename='Hg_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4364.44, compname='OIII', linename='OIII4363', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4862.68, compname='Hb', linename='Hb_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4687.02, compname='HeII', linename='HeII4686', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=4960.30, compname='OIII', linename='OIII4959', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=32, fvalue=1.0),
    _line_row(lam=5008.24, compname='OIII', linename='OIII5007', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=32, fvalue=_lnlam_peak_ratio_for_flux_ratio(2.98, 5008.24, 4960.30)),
    _line_row(lam=5877.25, compname='HeI', linename='HeI5876', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=6302.05, compname='OI', linename='OI6300', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=33, fvalue=_lnlam_peak_ratio_for_flux_ratio(3.05, 6302.05, 6365.54)),
    _line_row(lam=6365.54, compname='OI', linename='OI6363', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=33, fvalue=1.0),
    _line_row(lam=6549.85, compname='NII', linename='NII6548', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=34, fvalue=1.0),
    _line_row(lam=6564.61, compname='Ha', linename='Ha_na_elg', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=6585.28, compname='NII', linename='NII6583', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=34, fvalue=_lnlam_peak_ratio_for_flux_ratio(3.0, 6585.28, 6549.85)),
    _line_row(lam=6718.29, compname='SII', linename='SII6716', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=35, fvalue=1.0),
    _line_row(lam=6732.67, compname='SII', linename='SII6731', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=35, fvalue=1.0),
    # Red optical / far-red forbidden + He I
    _line_row(lam=7067.17, compname='HeI', linename='HeI7065', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=7137.77, compname='ArIII', linename='ArIII7138', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    _line_row(lam=7322.19, compname='OII', linename='OII7320', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=22, fvalue=0.001),
    _line_row(lam=7332.97, compname='OII', linename='OII7330', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=22, fvalue=0.001),
    _line_row(lam=7753.19, compname='ArIII', linename='ArIII7751', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=0, fvalue=0.001),
    # Higher-order Paschen series (vacuum wavelengths, narrow by default).
    # The stronger Pa-epsilon through Pa-alpha lines are part of the default
    # AGN table above.
    _line_row(lam=8752.87, compname='Paschen', linename='Pa12', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=8865.22, compname='Paschen', linename='Pa11', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=9017.38, compname='Paschen', linename='Pa10', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=9231.55, compname='Paschen', linename='Pa9', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    # Strong red/NIR forbidden lines
    _line_row(lam=9071.09, compname='SIII', linename='SIII9069', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=23, fvalue=1.0),
    _line_row(lam=9533.20, compname='SIII', linename='SIII9531', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=11, windex=11, findex=23, fvalue=_lnlam_peak_ratio_for_flux_ratio(2.5, 9533.20, 9071.09)),
]

# Optional high-ionization/coronal narrow-line set.
DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    _line_row(lam=3346.79, compname='NeV', linename='NeV3346', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=12, windex=12, findex=41, fvalue=1.0),
    _line_row(lam=3426.84, compname='NeV', linename='NeV3426_hi', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg, vindex=12, windex=12, findex=41, fvalue=_lnlam_peak_ratio_for_flux_ratio(2.7, 3426.84, 3346.79)),
    _line_row(lam=5721.0, compname='FeVII', linename='FeVII5721', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=6087.0, compname='FeVII', linename='FeVII6087', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
    _line_row(lam=6374.0, compname='FeX', linename='FeX6374', inisig=inisig_narrow, minsig=minsig_narrow, maxsig=maxsig_narrow, voff=voff_elg_red, vindex=12, windex=12, findex=0, fvalue=0.001),
]


def _apply_robust_line_scale_priors(
    line_rows: List[Dict[str, Any]],
    fscale: float,
    fmax: float,
) -> List[Dict[str, Any]]:
    """Apply flux-aware robust bounds/initialization to line-scale priors.

    Parameters
    ----------
    line_rows : object
        line_rows value.
    fscale : object
        fscale value.
    fmax : object
        fmax value.
    """
    if len(line_rows) == 0:
        return line_rows

    # Keep dynamic range positive even for nearly flat/noisy spectra.
    delta = max(float(fmax - fscale), 0.1 * float(fscale), AMPLITUDE_FLOOR)

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

        # Keep scales strictly positive and ordered.  Default qsopar rows use
        # ``inisca=0`` as a sentinel, but initializing a bounded amplitude at
        # the resulting lower floor leaves its unconstrained coordinate deep
        # in the transform tail.  Optax can then fail to move an otherwise
        # strong line away from zero.  Put sentinel/non-finite starts safely
        # inside the data-scaled interval while preserving explicit positive
        # user initializations.
        mins_floor = max(minsca, 1e-4 * float(fscale), AMPLITUDE_FLOOR)
        maxsca = max(maxsca, 1.01 * mins_floor)
        init_fraction = 0.05 if is_broad else 0.02
        if not np.isfinite(inisca) or inisca <= mins_floor:
            inisca = mins_floor + init_fraction * (maxsca - mins_floor)
        init_margin = max(1e-6 * (maxsca - mins_floor), AMPLITUDE_FLOOR)
        inisca = float(np.clip(inisca, mins_floor + init_margin, maxsca - init_margin))

        row["minsca"] = mins_floor
        row["maxsca"] = maxsca
        row["inisca"] = inisca

    return line_rows


def _append_unique_by_wavelength(
    base_rows: List[Dict[str, Any]],
    extra_rows: List[Dict[str, Any]],
    atol_angstrom: float = 1.0,
) -> List[Dict[str, Any]]:
    """Append rows from `extra_rows` only if no near-duplicate wavelength exists.

    Parameters
    ----------
    base_rows : object
        base_rows value.
    extra_rows : object
        extra_rows value.
    atol_angstrom : object
        atol_angstrom value.
    """
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


def append_optional_line_rows(
    prior_config: Dict[str, Any],
    flux: np.ndarray,
    *,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
) -> Dict[str, Any]:
    """Append optional built-in line sets selected by ``LineConfig``.

    Existing rows win when an optional row has the same wavelength, so this
    preserves user-provided line definitions and avoids duplicate components.
    Newly appended rows receive the same data-scaled amplitude initialization
    and bounds as rows constructed by the default-prior builder.
    """
    line_config = prior_config.get("line", {})
    if not isinstance(line_config, dict):
        return prior_config
    table = line_config.get("table")
    if not isinstance(table, list):
        return prior_config

    extras: List[Dict[str, Any]] = []
    if include_elg_narrow_lines:
        extras.extend(copy.deepcopy(DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS))
    if include_high_ionization_lines:
        extras.extend(copy.deepcopy(DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS))
    if not extras:
        return prior_config

    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    fmax = (
        float(np.nanpercentile(np.abs(f[finite]), ROBUST_FLUX_HIGH_PERCENTILE))
        if np.any(finite)
        else fscale
    )
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = fscale
    extras = _apply_robust_line_scale_priors(extras, fscale=fscale, fmax=fmax)
    line_config["table"] = _append_unique_by_wavelength(
        list(table), extras, atol_angstrom=1.0
    )
    return prior_config


def build_default_bal_components(
    flux: np.ndarray,
    *,
    tau_scale: float = 0.25,
    covering_loc: float = 0.15,
    covering_scale: float = 0.12,
    covering_high: float = 0.70,
    fwhm_kms_loc: float = 8000.0,
    fwhm_kms_scale: float = 2500.0,
    fwhm_kms_low: float = 2000.0,
    fwhm_kms_high: float = 15000.0,
) -> tuple[CustomComponentSpec, ...]:
    """Return built-in BAL custom components with conservative depth priors.

    Parameters
    ----------
    flux : object
        flux value.
    tau_scale : object
        tau_scale value.
    covering_loc : object
        covering_loc value.
    covering_scale : object
        covering_scale value.
    covering_high : object
        covering_high value.
    fwhm_kms_loc : object
        fwhm_kms_loc value.
    fwhm_kms_scale : object
        fwhm_kms_scale value.
    fwhm_kms_low : object
        fwhm_kms_low value.
    fwhm_kms_high : object
        fwhm_kms_high value.
    """
    def _bal_component(
        name: str,
        tau_scale: float,
        line_lambda: float,
        v_out_loc: float,
        v_out_scale: float,
        v_out_low: float,
        v_out_high: float,
    ):
        """Build one multiplicative BAL optical-depth component spec.

        Parameters
        ----------
        name : object
            name value.
        tau_scale : object
            tau_scale value.
        line_lambda : object
            line_lambda value.
        v_out_loc : object
            v_out_loc value.
        v_out_scale : object
            v_out_scale value.
        v_out_low : object
            v_out_low value.
        v_out_high : object
            v_out_high value.
        """
        return make_custom_component(
            name=name,
            parameter_priors={
                "tau_peak": dist.HalfNormal(float(max(tau_scale, 1.0e-6))),
                "covering": dist.TruncatedNormal(
                    float(covering_loc), float(max(covering_scale, 1.0e-6)),
                    low=0.0, high=float(covering_high),
                ),
                "v_out": dist.TruncatedNormal(
                    # The center is computed as lambda0 * (1 - v_out / c), so
                    # positive v_out values force absorption blueward of the
                    # associated transition.
                    float(v_out_loc), float(v_out_scale),
                    low=float(v_out_low), high=float(v_out_high),
                ),
                "fwhm_kms": dist.TruncatedNormal(
                    float(fwhm_kms_loc), float(max(fwhm_kms_scale, 1.0e-6)),
                    low=float(fwhm_kms_low), high=float(fwhm_kms_high),
                ),
                "shape_power": dist.TruncatedNormal(2.0, 1.5, low=2.0, high=12.0),
            },
            evaluate=gaussian_bal_optical_depth_component,
            metadata={
                "component_type": "bal_absorption",
                "line_lambda": float(line_lambda),
                "shared_parameter_sites": {
                    "v_out": "custom_bal_v_out",
                    "tau_peak": "custom_bal_tau_peak",
                    "covering": "custom_bal_covering",
                    "fwhm_kms": "custom_bal_fwhm_kms",
                },
            },
        )

    # Trump et al. (2006)
    return (
        _bal_component("bal_nv", tau_scale=tau_scale, line_lambda=1240.14, v_out_loc=6000.0, v_out_scale=2500.0, v_out_low=3000.0, v_out_high=12000.0),
        # _bal_component("bal_nv_2", depth_frac=0.025, center=1160.0, scale=90.0, low=1100.0, high=1240.0, sigma=40.0),
        _bal_component("bal_siiv", tau_scale=tau_scale, line_lambda=1396.76, v_out_loc=6000.0, v_out_scale=2500.0, v_out_low=3000.0, v_out_high=12000.0),
        # _bal_component("bal_siiv_2", depth_frac=0.025, center=1320.0, scale=90.0, low=1260.0, high=1397.0, sigma=40.0),
        _bal_component("bal_civ", tau_scale=tau_scale, line_lambda=1549.06, v_out_loc=6000.0, v_out_scale=2500.0, v_out_low=3000.0, v_out_high=12000.0),
        # _bal_component("bal_civ_2", depth_frac=0.03, center=1450.0, scale=100.0, low=1350.0, high=1549.0, sigma=45.0),
        # not common, often blended with other lines
        # _bal_component("bal_ciii", tau_scale=0.8, line_lambda=1908.73, v_out_loc=9200.0, v_out_scale=8000.0, v_out_low=300.0, v_out_high=25000.0, sigma=30.0),
        # _bal_component("bal_ciii_2", depth_frac=0.02, center=1800.0, scale=100.0, low=1700.0, high=1909.0, sigma=50.0),
        # Fe absorption, not common
        # _bal_component("bal_fe1", tau_scale=0.8, line_lambda=2050.0, v_out_loc=7300.0, v_out_scale=8000.0, v_out_low=300.0, v_out_high=15000.0, sigma=30.0),
        # _bal_component("bal_fe2", tau_scale=0.8, line_lambda=2250.0, v_out_loc=6600.0, v_out_scale=8000.0, v_out_low=300.0, v_out_high=13000.0, sigma=30.0),
        # not common
        # _bal_component("bal_mgii", tau_scale=0.8, line_lambda=2798.75, v_out_loc=5000.0, v_out_scale=7000.0, v_out_low=300.0, v_out_high=5200.0, sigma=40.0),
        # _bal_component("bal_mgii_2", depth_frac=0.02, center=2760.0, scale=120.0, low=2700.0, high=2798.0, sigma=55.0),
    )


def _build_default_prior_config(
    flux: np.ndarray,
    line_config: Dict[str, Any] | None = None,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
    pl_pivot: float | None = None,
) -> Dict[str, Any]:
    """Build a full PriorConfig with sane defaults from data flux scale.

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

    Notes
    -----
    ``log_ebv`` controls the amplitude of the built-in SMC-like attenuation
    curve in log space and is literal :math:`E(B-V)=A_B-A_V`. Legacy
    ``log_reddening_a2500`` prior dictionaries remain supported.
    """
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    fmax = (
        float(np.nanpercentile(np.abs(f[finite]), ROBUST_FLUX_HIGH_PERCENTILE))
        if np.any(finite)
        else fscale
    )
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = fscale

    cfg: Dict[str, Any] = {
        "cont_norm": dist.LogNormal(np.log(max(fscale, AMPLITUDE_FLOOR)), 0.3),
        "PL_norm": dist.HalfNormal(max(0.5 * fscale, AMPLITUDE_FLOOR)),
        "PL_slope": dist.Normal(-1.5, 0.4),
        "PL_pivot": None if pl_pivot is None else float(pl_pivot),
        "poly_pivot": None,
        # This corresponds to the historical median A(2500)=0.1 mag.
        "log_ebv": dist.Normal(np.log(0.1 * ((4400.0 / 2500.0) ** -1.2 - (5500.0 / 2500.0) ** -1.2)), 0.6),
        "reddening_uv_ref": 2500.0,
        "reddening_alpha": 1.2,
        "residualize_reddening_geometry": True,
        "log_frac_host": dist.StudentT(df=3.0, loc=0.0, scale=2.0),
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
        "tau_host": dist.HalfNormal(1.0),
        "raw_w": dist.Normal(-0.5, 1.0),
        "host_template_age_prior": {
            "type": "prefer_old",
            "pivot_gyr": 1.0,
            "strength": 1.0,
            "min_logit": -3.0,
            "max_logit": 2.0,
        },
        "log_stellar_mass": dist.TruncatedNormal(9.0, 0.75, low=7.0, high=12.0),
        "log_host_aperture_scale": dist.Normal(0.0, 0.5),
        "log_sfh_age_gyr": dist.Normal(np.log(3.0), 1.0),
        "log_sfh_tau_over_age": dist.Normal(0.0, 0.5),
        "gal_lgmet": dist.Normal(0.0, 0.5),
        "log_gal_lgmet_scatter": dist.Normal(np.log(0.15), 0.7),
        "mass_metallicity_relation": {
            "enabled": False,
            "pivot_mass": 10.0,
            "pivot_logzsol": -0.15,
            "slope": 0.35,
            "scale": 0.25,
            "min": -1.5,
            "max": 0.3,
        },
        "gal_v_kms": dist.Normal(0.0, 120.0),
        "log_gal_sigma_kms": dist.TruncatedNormal(np.log(150.0), 0.4, low=np.log(30.0), high=np.log(500.0)),
        "Fe_uv_norm": dist.LogNormal(np.log(max(0.03 * fscale, 1e-12)), 1.0),
        "log_Fe_op_over_uv": dist.Normal(0.0, 1.0),
        "Fe_FWHM": dist.LogNormal(np.log(3000.0), 0.5),
        "Fe_shift": dist.Normal(0.0, 1e-3),
        "Balmer_norm": dist.LogNormal(np.log(max(1e-3 * fscale, AMPLITUDE_FLOOR)), 0.5),
        "Balmer_Tau": dist.LogNormal(np.log(0.5), 0.25),
        "log_Balmer_vel": dist.TruncatedNormal(np.log(3000.0), 0.3, low=np.log(1000.0), high=np.log(15000.0)),
        "poly_c2": dist.Normal(0.0, 0.03),
        "poly_c3": dist.Normal(0.0, 0.03),
        "poly_c4": dist.Normal(0.0, 0.03),
        "poly_c5": dist.Normal(0.0, 0.03),
        "poly_c6": dist.Normal(0.0, 0.03),
        "frac_jitter": dist.HalfNormal(0.02),
        "frac_fe_jitter": {"dist": "Delta", "value": 0.20},
        "add_jitter": {"dist": "Delta", "value": 0.0},
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
    return PriorConfig._from_model_priors(cfg)
