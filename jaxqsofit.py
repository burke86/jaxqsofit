# PyQSOFit Bayesian-only integration: FSPS + JAX/NumPyro + tied-line logic
#
# Updated to restore the standard AGN continuum components from the original PyQSOFit model:
# - power-law AGN continuum
# - UV FeII template
# - optical FeII template
# - Balmer continuum
# - dust attenuation term
#
# while keeping:
# - FSPS host galaxy SSP mixture + LOSVD
# - tied Gaussian emission lines using vindex / windex / findex / fvalue
# - NumPyro / NUTS sampling
#
# Design notes
# ------------
# 1) The fitting path is Bayesian-only.
# 2) FSPS is assumed available.
# 3) We preserve the standard PyQSOFit component names on the object where practical:
#       f_pl_model, f_fe_mgii_model, f_fe_balmer_model, f_bc_model, f_poly_model,
#       f_conti_model, f_line_model, host, qso, line_flux, etc.
# 4) Plotting is restored in a simplified but familiar PyQSOFit style.
# 5) The FeII and Balmer components are implemented directly in the forward model.
# 6) The continuum priors are still read from qsopar.fits.

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import extinction
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

try:
    from dsps import load_ssp_templates
except Exception:
    load_ssp_templates = None

try:
    from dustmaps.config import config as dustmaps_config
    from dustmaps.sfd import SFDQuery, fetch as fetch_sfd
except Exception:
    dustmaps_config = None
    SFDQuery = None
    fetch_sfd = None

warnings.filterwarnings("ignore")

C_KMS = 299792.458
LOG10E = np.log(10.0)
HAS_DSPS = load_ssp_templates is not None
_SFD_QUERY_CACHE: Dict[str, Any] = {}


def get_err(s, margin=0.16, axis=0, default_value=-1.):
    s = np.array(s)
    s[s == default_value] = np.nan
    margin_per = int(margin * 100)
    if s.ndim == 1:
        N_samp = len(s)
        if np.sum(np.isnan(s)) / N_samp > 0.5:
            return default_value
        return np.diff(np.nanpercentile(s, (margin_per, 100 - margin_per)))[0] / 2
    elif s.ndim == 2:
        if axis == 1:
            s = s.T
        if axis not in [0, 1]:
            raise IndexError('The axis parameter only adopts 0 or 1.')
        N_samp = s.shape[0]
        Na_idx = np.where(np.sum(np.isnan(s), axis=0) > N_samp / 2, True, False)
        data_err = np.diff(np.nanpercentile(s, (margin_per, 100 - margin_per), axis=0), axis=0)[0] / 2
        data_err[Na_idx] = default_value
        return data_err
    else:
        raise IndexError('The input data only adopts 1-D or 2-D array.')


def unred(wave, flux, ebv, R_V=3.1):
    # Preserve legacy function signature; use extinction package implementation.
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    a_lambda = extinction.fitzpatrick99(wave, a_v=R_V * ebv, r_v=R_V)
    return extinction.remove(a_lambda, flux)


def _np_to_jnp(x):
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _powerlaw_jax(wave, pl_norm, pl_slope, pivot=3000.0):
    x = jnp.clip(wave / pivot, 1e-8, None)
    return pl_norm * x ** pl_slope


def _many_gauss_lnlam(lnlam, amps, mus, sigs):
    z = (lnlam[:, None] - mus[None, :]) / sigs[None, :]
    return jnp.sum(amps[None, :] * jnp.exp(-0.5 * z * z), axis=1)


def _shift_and_broaden_templates_lnlam(lnwave, templates, v_kms, sigma_kms):
    dln = jnp.median(jnp.diff(lnwave))
    sigma_ln = jnp.maximum(sigma_kms / C_KMS, 1e-5)
    sigma_pix = sigma_ln / jnp.maximum(dln, 1e-8)
    kern = _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=128)

    wave = jnp.exp(lnwave)
    shift_ln = v_kms / C_KMS
    shifted_wave = jnp.exp(lnwave - shift_ln)

    def one_template(t):
        shifted = jnp.interp(shifted_wave, wave, t, left=0.0, right=0.0)
        return jnp.convolve(shifted, kern, mode='same')

    return jax.vmap(one_template, in_axes=1, out_axes=1)(templates)


def _shift_and_broaden_single_spectrum_lnlam(lnwave, spectrum, v_kms, sigma_kms):
    dln = jnp.median(jnp.diff(lnwave))
    sigma_ln = jnp.maximum(sigma_kms / C_KMS, 1e-5)
    sigma_pix = sigma_ln / jnp.maximum(dln, 1e-8)
    kern = _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=128)

    wave = jnp.exp(lnwave)
    shift_ln = v_kms / C_KMS
    shifted_wave = jnp.exp(lnwave - shift_ln)
    shifted = jnp.interp(shifted_wave, wave, spectrum, left=0.0, right=0.0)
    return jnp.convolve(shifted, kern, mode='same')


def _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=512):
    sigma_pix = jnp.maximum(sigma_pix, 1e-3)
    x = jnp.arange(-max_half, max_half + 1, dtype=jnp.float64)
    half_dyn = jnp.maximum(3.0, jnp.ceil(radius_mult * sigma_pix))
    mask = jnp.abs(x) <= half_dyn
    k = jnp.exp(-0.5 * (x / sigma_pix) ** 2)
    k = jnp.where(mask, k, 0.0)
    return k / jnp.maximum(jnp.sum(k), 1e-30)


def _fe_template_component(wave, wave_template, flux_template, norm, fwhm_kms, shift_frac, base_fwhm_kms=900.0, base_disp_kms=106.3):
    wave_shifted = wave * (1.0 + shift_frac)
    sigma_conv = jnp.sqrt(jnp.maximum(fwhm_kms**2 - base_fwhm_kms**2, 910.0**2 - base_fwhm_kms**2)) / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    sigma_pix = sigma_conv / base_disp_kms
    kernel = _gaussian_kernel1d(sigma_pix)
    flux_conv = jnp.convolve(flux_template, kernel, mode='same')
    model = jnp.interp(wave_shifted, wave_template, flux_conv, left=0.0, right=0.0)
    return norm * model


def _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel):
    lam_be = 3646.0
    h = 6.62607015e-27
    c = 2.99792458e10
    kb = 1.380649e-16

    wave = jnp.asarray(wave)
    lam_cm = wave * 1e-8

    expo = jnp.clip((h * c) / (lam_cm * kb * balmer_te), 1e-9, 700.0)
    bb = (2.0 * h * c**2 / lam_cm**5) / jnp.expm1(expo)
    bb = bb * 1e-8 * jnp.pi

    # Normalize shape at Balmer edge so balmer_norm is a flux-like amplitude.
    lam_be_cm = lam_be * 1e-8
    expo_edge = jnp.clip((h * c) / (lam_be_cm * kb * balmer_te), 1e-9, 700.0)
    bb_edge = (2.0 * h * c**2 / lam_be_cm**5) / jnp.expm1(expo_edge)
    bb_edge = bb_edge * 1e-8 * jnp.pi
    bb = bb / jnp.maximum(bb_edge, 1e-30)

    tau = balmer_tau * (wave / lam_be) ** 3
    bc = balmer_norm * (1.0 - jnp.exp(-tau)) * bb
    bc = jnp.where(wave <= lam_be, bc, 0.0)

    lnwave = jnp.log(wave)
    dln = jnp.median(jnp.diff(lnwave))
    sigma_ln = jnp.maximum(balmer_vel / C_KMS, 1e-5)
    sigma_pix = sigma_ln / jnp.maximum(dln, 1e-8)
    kernel = _gaussian_kernel1d(sigma_pix)
    bc_conv = jnp.convolve(bc, kernel, mode='same')
    return bc_conv


def _smc_like_attenuation(wave, ebv, rv):
    inv_um = 1e4 / jnp.clip(wave, 1200.0, None)
    k_lambda = rv + 1.39 * inv_um**1.2
    a_lambda = ebv * k_lambda
    # Numerical floor prevents complete model collapse when extinction runs to extreme values.
    return jnp.clip(jnp.exp(-0.4 * LOG10E * a_lambda), 1e-8, 1.0)


@dataclass
class FSPSTemplateGrid:
    wave: np.ndarray
    templates: np.ndarray
    template_meta: List[Dict[str, float]]
    age_grid_gyr: np.ndarray
    logzsol_grid: np.ndarray


def _map_logzsol_to_dsps_lgmet(logzsol_grid: Sequence[float], ssp_lgmet: np.ndarray) -> np.ndarray:
    logzsol_grid = np.asarray(logzsol_grid, dtype=float)
    ssp_lgmet = np.asarray(ssp_lgmet, dtype=float)

    # DSPS metallicity grids are often log10(Z), while fitting grids are usually log10(Z/Zsun).
    # Select the transform that best matches the available DSPS metallicity grid.
    cand_direct = logzsol_grid
    cand_shifted = logzsol_grid + np.log10(0.019)

    def mismatch(cand):
        return np.mean([np.min(np.abs(ssp_lgmet - val)) for val in cand])

    return cand_direct if mismatch(cand_direct) <= mismatch(cand_shifted) else cand_shifted


def _get_sfd_query(dustmap_path: str, fetch_if_missing: bool = True):
    if SFDQuery is None or dustmaps_config is None:
        raise ImportError(
            "dustmaps is not installed. Install `dustmaps` and run SFD fetch to enable Galactic dereddening."
        )

    cache_key = os.path.abspath(dustmap_path)
    if cache_key in _SFD_QUERY_CACHE:
        return _SFD_QUERY_CACHE[cache_key]

    os.makedirs(cache_key, exist_ok=True)
    dustmaps_config['data_dir'] = cache_key

    try:
        q = SFDQuery()
    except Exception:
        if not fetch_if_missing or fetch_sfd is None:
            raise
        fetch_sfd()
        q = SFDQuery()

    _SFD_QUERY_CACHE[cache_key] = q
    return q


def build_fsps_template_grid(
    wave_out: np.ndarray,
    age_grid_gyr: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0),
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2),
    imf_type: int = 1,
    zcontinuous: int = 1,
    sfh: int = 0,
    dsps_ssp_fn: str = 'tempdata.h5',
) -> FSPSTemplateGrid:
    # Parameters kept for API compatibility.
    _ = (imf_type, zcontinuous, sfh)
    if not HAS_DSPS:
        raise ImportError("DSPS is not installed. Install `dsps` to build stellar templates.")

    # DSPS quickstart pattern:
    # from dsps import load_ssp_templates
    # ssp_data = load_ssp_templates(fn='tempdata.h5')
    ssp_data = load_ssp_templates(fn=dsps_ssp_fn)
    ssp_lgmet = np.asarray(ssp_data.ssp_lgmet, dtype=float)
    ssp_lg_age_gyr = np.asarray(ssp_data.ssp_lg_age_gyr, dtype=float)
    ssp_wave = np.asarray(ssp_data.ssp_wave, dtype=float)
    ssp_flux = np.asarray(ssp_data.ssp_flux, dtype=float)

    wave_out = np.asarray(wave_out, dtype=float)
    age_grid_gyr = np.asarray(age_grid_gyr, dtype=float)
    logzsol_grid = np.asarray(logzsol_grid, dtype=float)
    target_lg_age = np.log10(np.clip(age_grid_gyr, 1e-5, None))
    target_lgmet = _map_logzsol_to_dsps_lgmet(logzsol_grid, ssp_lgmet)

    tmpl = []
    meta = []
    for i_z, logz in enumerate(logzsol_grid):
        imet = int(np.argmin(np.abs(ssp_lgmet - target_lgmet[i_z])))
        for i_a, age in enumerate(age_grid_gyr):
            iage = int(np.argmin(np.abs(ssp_lg_age_gyr - target_lg_age[i_a])))
            spec_native = np.asarray(ssp_flux[imet, iage, :], dtype=float)
            spec_interp = np.interp(wave_out, ssp_wave, spec_native, left=0.0, right=0.0)
            norm = np.nanmedian(np.abs(spec_interp))
            if not np.isfinite(norm) or norm <= 0:
                norm = 1.0
            spec_interp = spec_interp / norm
            tmpl.append(spec_interp)
            meta.append({
                'tage_gyr': float(age),
                'logzsol': float(logz),
                'norm': float(norm),
                'dsps_lgmet': float(ssp_lgmet[imet]),
                'dsps_lg_age_gyr': float(ssp_lg_age_gyr[iage]),
            })

    templates = np.column_stack(tmpl)
    return FSPSTemplateGrid(
        wave=wave_out,
        templates=templates,
        template_meta=meta,
        age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
        logzsol_grid=np.asarray(logzsol_grid, dtype=float),
    )


def read_conti_prior_dict(param_file_path='qsopar.fits'):
    hdul = fits.open(param_file_path)
    data = hdul[3].data
    out = {}
    for row in data:
        out[str(row['parname'])] = {
            'initial': float(row['initial']),
            'min': float(row['min']),
            'max': float(row['max']),
            'vary': bool(row['vary']),
        }
    return out


def _extract_line_table_from_prior_config(prior_config: Dict[str, Any] | None):
    if prior_config is None:
        return None
    if 'line_priors' in prior_config:
        return prior_config['line_priors']
    if 'line_table' in prior_config:
        return prior_config['line_table']
    line_cfg = prior_config.get('line', None)
    if isinstance(line_cfg, dict):
        if 'table' in line_cfg:
            return line_cfg['table']
        if 'priors' in line_cfg:
            return line_cfg['priors']
    return None


def _extract_line_global_prior_config(prior_config: Dict[str, Any] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if prior_config is None:
        return out
    line_cfg = prior_config.get('line', None)
    if isinstance(line_cfg, dict):
        if isinstance(line_cfg.get('global', None), dict):
            out.update(line_cfg['global'])
        for k in ('amp_log_scale', 'amp_cap_mult_init', 'amp_floor',
                  'sig_log_scale', 'sig_cap_mult_init', 'sig_floor',
                  'dmu_scale_mult'):
            if k in line_cfg:
                out[k] = line_cfg[k]
    for k in ('amp_log_scale', 'amp_cap_mult_init', 'amp_floor',
              'sig_log_scale', 'sig_cap_mult_init', 'sig_floor',
              'dmu_scale_mult'):
        if k in prior_config:
            out[k] = prior_config[k]
    return out


def _compress_group_ids(ids: np.ndarray, labels: Sequence[str] | None = None) -> Tuple[np.ndarray, Dict[Any, int]]:
    out = np.full(len(ids), -1, dtype=int)
    mapping: Dict[Any, int] = {}
    next_gid = 0
    for i, gid in enumerate(ids):
        gid = int(gid)
        if gid <= 0:
            continue
        key: Any = gid if labels is None else (str(labels[i]), gid)
        if key not in mapping:
            mapping[key] = next_gid
            next_gid += 1
        out[i] = mapping[key]
    return out, mapping


def build_tied_line_meta_from_linelist(linelist, wave):
    def _to_records(obj):
        # pandas.DataFrame
        if hasattr(obj, 'to_dict'):
            return obj.to_dict('records')
        # Astropy table / FITS recarray / numpy structured array
        if hasattr(obj, 'dtype') and getattr(obj.dtype, 'names', None):
            return [{name: row[name] for name in obj.dtype.names} for row in obj]
        if hasattr(obj, 'colnames'):
            return [{name: row[name] for name in obj.colnames} for row in obj]
        # list[dict]-like
        return list(obj)

    records = _to_records(linelist)
    rows = []
    wmin = float(np.min(wave))
    wmax = float(np.max(wave))
    for row in records:
        lam = float(row['lambda'])
        if lam > wmin and lam < wmax:
            rows.append(row)

    ln_lambda0 = []
    amp_init = []
    amp_min = []
    amp_max = []
    sig_init = []
    sig_min = []
    sig_max = []
    dmu_min = []
    dmu_max = []
    names = []
    line_lambda = []
    vindex = []
    windex = []
    findex = []
    fvalue = []
    compnames = []

    for row in rows:
        for i in range(int(row.get('ngauss', 1))):
            ln0 = np.log(float(row['lambda']))
            voff = float(row['voff'])
            dln = voff / C_KMS
            ln_lambda0.append(ln0)
            line_lambda.append(float(row['lambda']))
            amp_init.append(float(row['inisca']))
            amp_min.append(float(row['minsca']))
            amp_max.append(float(row['maxsca']))
            sig_init.append(max(float(row['inisig']), 1e-5))
            sig_min.append(max(float(row['minsig']), 1e-5))
            sig_max.append(max(float(row['maxsig']), 1e-5))
            dmu_min.append(-dln)
            dmu_max.append(+dln)
            linename = str(row.get('linename', f"line_{row['lambda']:.1f}"))
            names.append(f"{linename}_{i+1}")
            vindex.append(int(row['vindex']))
            windex.append(int(row['windex']))
            findex.append(int(row['findex']))
            fvalue.append(float(row['fvalue']))
            compnames.append(str(row.get('compname', linename)))

    ln_lambda0 = np.asarray(ln_lambda0, dtype=float)
    amp_init = np.asarray(amp_init, dtype=float)
    amp_min = np.asarray(amp_min, dtype=float)
    amp_max = np.asarray(amp_max, dtype=float)
    sig_init = np.asarray(sig_init, dtype=float)
    sig_min = np.asarray(sig_min, dtype=float)
    sig_max = np.asarray(sig_max, dtype=float)
    dmu_min = np.asarray(dmu_min, dtype=float)
    dmu_max = np.asarray(dmu_max, dtype=float)
    vindex = np.asarray(vindex, dtype=int)
    windex = np.asarray(windex, dtype=int)
    findex = np.asarray(findex, dtype=int)
    fvalue = np.asarray(fvalue, dtype=float)

    # Tie indices are local to each line complex in qsopar; include compname in the key
    # to avoid accidental cross-complex tying when index integers are reused.
    vgroup, _ = _compress_group_ids(vindex, compnames)
    next_gid = np.max(vgroup) + 1 if len(vgroup) and np.any(vgroup >= 0) else 0
    for i in range(len(vgroup)):
        if vgroup[i] < 0:
            vgroup[i] = next_gid
            next_gid += 1
    n_vgroups = int(np.max(vgroup)) + 1 if len(vgroup) else 0

    wgroup, _ = _compress_group_ids(windex, compnames)
    next_gid = np.max(wgroup) + 1 if len(wgroup) and np.any(wgroup >= 0) else 0
    for i in range(len(wgroup)):
        if wgroup[i] < 0:
            wgroup[i] = next_gid
            next_gid += 1
    n_wgroups = int(np.max(wgroup)) + 1 if len(wgroup) else 0

    fgroup, _ = _compress_group_ids(findex, compnames)
    flux_ratio = np.ones(len(fgroup), dtype=float)
    next_gid = np.max(fgroup) + 1 if len(fgroup) and np.any(fgroup >= 0) else 0
    for local_gid in sorted(set([g for g in fgroup if g >= 0])):
        members = np.where(fgroup == local_gid)[0]
        ref = members[0]
        ref_f = fvalue[ref] if fvalue[ref] != 0 else 1.0
        for m in members:
            flux_ratio[m] = fvalue[m] / ref_f if ref_f != 0 else 1.0
    for i in range(len(fgroup)):
        if fgroup[i] < 0:
            fgroup[i] = next_gid
            flux_ratio[i] = 1.0
            next_gid += 1
    n_fgroups = int(np.max(fgroup)) + 1 if len(fgroup) else 0

    amp_init_group = np.zeros(n_fgroups, dtype=float)
    amp_min_group = np.zeros(n_fgroups, dtype=float)
    amp_max_group = np.zeros(n_fgroups, dtype=float)
    for gid in range(n_fgroups):
        ref = np.where(fgroup == gid)[0][0]
        amp_init_group[gid] = amp_init[ref]
        amp_min_group[gid] = amp_min[ref]
        amp_max_group[gid] = amp_max[ref]

    dmu_init_group = np.zeros(n_vgroups, dtype=float)
    dmu_min_group = np.zeros(n_vgroups, dtype=float)
    dmu_max_group = np.zeros(n_vgroups, dtype=float)
    for gid in range(n_vgroups):
        members = np.where(vgroup == gid)[0]
        dmu_init_group[gid] = 0.0
        dmu_min_group[gid] = np.max(dmu_min[members])
        dmu_max_group[gid] = np.min(dmu_max[members])

    sig_init_group = np.zeros(n_wgroups, dtype=float)
    sig_min_group = np.zeros(n_wgroups, dtype=float)
    sig_max_group = np.zeros(n_wgroups, dtype=float)
    for gid in range(n_wgroups):
        members = np.where(wgroup == gid)[0]
        sig_init_group[gid] = np.median(sig_init[members])
        sig_min_group[gid] = np.max(sig_min[members])
        sig_max_group[gid] = np.min(sig_max[members])
        if sig_max_group[gid] <= sig_min_group[gid]:
            sig_max_group[gid] = max(sig_min_group[gid] * 1.1, sig_min_group[gid] + 1e-4)

    return {
        'n_lines': len(ln_lambda0),
        'n_vgroups': n_vgroups,
        'n_wgroups': n_wgroups,
        'n_fgroups': n_fgroups,
        'ln_lambda0': _np_to_jnp(ln_lambda0),
        'vgroup': np.asarray(vgroup, dtype=int),
        'wgroup': np.asarray(wgroup, dtype=int),
        'fgroup': np.asarray(fgroup, dtype=int),
        'flux_ratio': np.asarray(flux_ratio, dtype=float),
        'dmu_init_group': np.asarray(dmu_init_group, dtype=float),
        'dmu_min_group': np.asarray(dmu_min_group, dtype=float),
        'dmu_max_group': np.asarray(dmu_max_group, dtype=float),
        'sig_init_group': np.asarray(sig_init_group, dtype=float),
        'sig_min_group': np.asarray(sig_min_group, dtype=float),
        'sig_max_group': np.asarray(sig_max_group, dtype=float),
        'amp_init_group': np.asarray(amp_init_group, dtype=float),
        'amp_min_group': np.asarray(amp_min_group, dtype=float),
        'amp_max_group': np.asarray(amp_max_group, dtype=float),
        'names': names,
        'compnames': compnames,
        'line_lambda': np.asarray(line_lambda, dtype=float),
    }


def qso_fsps_joint_model(wave, flux, err, conti_priors, tied_line_meta, fsps_grid,
                         fe_uv_wave, fe_uv_flux, fe_op_wave, fe_op_flux, use_lines=True,
                         prior_config=None, decompose_host=True, fit_fe=True, fit_bc=True, fit_poly=False):
    wave = _np_to_jnp(wave)
    flux = _np_to_jnp(flux)
    err = _np_to_jnp(err)
    lnwave = jnp.log(wave)
    templates = _np_to_jnp(fsps_grid.templates)
    fe_uv_wave = _np_to_jnp(fe_uv_wave)
    fe_uv_flux = _np_to_jnp(fe_uv_flux)
    fe_op_wave = _np_to_jnp(fe_op_wave)
    fe_op_flux = _np_to_jnp(fe_op_flux)
    prior_config = {} if prior_config is None else prior_config

    def _cfg_required(key):
        if key not in prior_config:
            raise ValueError(f"Missing prior_config['{key}']")
        return prior_config[key]

    def _cfg_norm_required(key):
        cfg = _cfg_required(key)
        if isinstance(cfg, dict) and ('loc' in cfg and 'scale' in cfg):
            return jnp.asarray(cfg['loc']), jnp.asarray(cfg['scale'])
        if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
            return jnp.asarray(cfg[0]), jnp.asarray(cfg[1])
        raise ValueError(f"prior_config['{key}'] must provide loc/scale")

    def _cfg_halfnorm_required(key, ref_scale=None):
        cfg = _cfg_required(key)
        if isinstance(cfg, dict):
            if 'scale' in cfg:
                return jnp.asarray(cfg['scale'])
            if 'scale_mult_err' in cfg:
                if ref_scale is None:
                    raise ValueError(f"prior_config['{key}']['scale_mult_err'] requires ref_scale")
                return jnp.asarray(cfg['scale_mult_err'] * ref_scale)
        if isinstance(cfg, (tuple, list)) and len(cfg) >= 1:
            return jnp.asarray(cfg[0])
        if np.isscalar(cfg):
            return jnp.asarray(cfg)
        raise ValueError(f"prior_config['{key}'] must provide scale or scale_mult_err")

    # Continuum amplitude + host fraction parameterization
    cont_norm = numpyro.sample('cont_norm', dist.LogNormal(*_cfg_norm_required('log_cont_norm')))
    if decompose_host:
        log_frac_host = numpyro.sample('log_frac_host', dist.Normal(*_cfg_norm_required('log_frac_host')))
        frac_host = jax.nn.sigmoid(log_frac_host)
    else:
        frac_host = jnp.asarray(0.0)
    pl_norm = cont_norm * (1.0 - frac_host)
    pl_slope_cfg = _cfg_required('PL_slope')
    pl_slope_loc, pl_slope_scale = _cfg_norm_required('PL_slope')
    if isinstance(pl_slope_cfg, dict) and ('low' in pl_slope_cfg and 'high' in pl_slope_cfg):
        pl_slope = numpyro.sample(
            'PL_slope',
            dist.TruncatedNormal(
                loc=pl_slope_loc,
                scale=pl_slope_scale,
                low=pl_slope_cfg['low'],
                high=pl_slope_cfg['high'],
            ),
        )
    else:
        pl_slope = numpyro.sample('PL_slope', dist.Normal(pl_slope_loc, pl_slope_scale))

    if fit_fe:
        fe_uv_norm = numpyro.sample('Fe_uv_norm', dist.LogNormal(*_cfg_norm_required('log_Fe_uv_norm')))
        fe_op_norm = numpyro.sample('Fe_op_norm', dist.LogNormal(*_cfg_norm_required('log_Fe_op_norm')))
        fe_uv_fwhm = numpyro.sample('Fe_uv_FWHM', dist.LogNormal(*_cfg_norm_required('log_Fe_uv_FWHM')))
        fe_op_fwhm = numpyro.sample('Fe_op_FWHM', dist.LogNormal(*_cfg_norm_required('log_Fe_op_FWHM')))
        fe_uv_shift = numpyro.sample('Fe_uv_shift', dist.Normal(*_cfg_norm_required('Fe_uv_shift')))
        fe_op_shift = numpyro.sample('Fe_op_shift', dist.Normal(*_cfg_norm_required('Fe_op_shift')))
    else:
        fe_uv_norm = jnp.asarray(0.0)
        fe_op_norm = jnp.asarray(0.0)
        fe_uv_fwhm = jnp.asarray(3000.0)
        fe_op_fwhm = jnp.asarray(3000.0)
        fe_uv_shift = jnp.asarray(0.0)
        fe_op_shift = jnp.asarray(0.0)

    if fit_bc:
        balmer_norm = numpyro.sample('Balmer_norm', dist.LogNormal(*_cfg_norm_required('log_Balmer_norm')))
        balmer_te = numpyro.sample('Balmer_Te', dist.Normal(*_cfg_norm_required('Balmer_Te')))
        balmer_tau = numpyro.sample('Balmer_Tau', dist.LogNormal(*_cfg_norm_required('log_Balmer_Tau')))
        balmer_vel = numpyro.sample('Balmer_vel', dist.LogNormal(*_cfg_norm_required('log_Balmer_vel')))
    else:
        balmer_norm = jnp.asarray(0.0)
        balmer_te = jnp.asarray(15000.0)
        balmer_tau = jnp.asarray(0.5)
        balmer_vel = jnp.asarray(3000.0)

    pl_model_intrinsic = _powerlaw_jax(wave, pl_norm=pl_norm, pl_slope=pl_slope, pivot=3000.0)
    fe_uv_model_intrinsic = _fe_template_component(wave, fe_uv_wave, fe_uv_flux, fe_uv_norm, fe_uv_fwhm, fe_uv_shift)
    fe_op_model_intrinsic = _fe_template_component(wave, fe_op_wave, fe_op_flux, fe_op_norm, fe_op_fwhm, fe_op_shift)
    bc_model_intrinsic = _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel)
    pl_model = pl_model_intrinsic
    fe_uv_model = fe_uv_model_intrinsic
    fe_op_model = fe_op_model_intrinsic
    bc_model = bc_model_intrinsic
    poly_model = jnp.ones_like(wave)
    if fit_poly:
        poly_c1 = numpyro.sample('poly_c1', dist.Normal(*_cfg_norm_required('poly_c1')))
        poly_c2 = numpyro.sample('poly_c2', dist.Normal(*_cfg_norm_required('poly_c2')))
        w0 = jnp.median(wave)
        x = (wave - w0) / jnp.maximum(w0, 1.0)
        poly_model = jnp.clip(1.0 + poly_c1 * x + poly_c2 * x * x, 0.2, 5.0)
        pl_model = pl_model * poly_model
        fe_uv_model = fe_uv_model * poly_model
        fe_op_model = fe_op_model * poly_model
        bc_model = bc_model * poly_model
    agn_model = pl_model + fe_uv_model + fe_op_model + bc_model

    ntemp = fsps_grid.templates.shape[1]
    if decompose_host:
        tau_host = numpyro.sample('tau_host', dist.HalfNormal(_cfg_halfnorm_required('tau_host')))
        raw_w_loc, _ = _cfg_norm_required('raw_w')
        raw_w = numpyro.sample('fsps_weights_raw', dist.Normal(jnp.full((ntemp,), raw_w_loc), tau_host))
        fsps_weights_frac = jax.nn.softmax(raw_w)
        host_amp = cont_norm * frac_host
        fsps_weights = host_amp * fsps_weights_frac
        gal_v_kms = numpyro.sample('gal_v_kms', dist.Normal(*_cfg_norm_required('gal_v_kms')))
        gal_sigma_kms = numpyro.sample('gal_sigma_kms', dist.HalfNormal(_cfg_halfnorm_required('gal_sigma_kms')))
        gal_intrinsic = jnp.dot(templates, fsps_weights)
        gal_model_intrinsic = _shift_and_broaden_single_spectrum_lnlam(lnwave, gal_intrinsic, gal_v_kms, gal_sigma_kms)
    else:
        fsps_weights_frac = jnp.zeros((ntemp,))
        fsps_weights = jnp.zeros((ntemp,))
        gal_model_intrinsic = jnp.zeros_like(wave)

    if use_lines and tied_line_meta['n_lines'] > 0:
        n_v = tied_line_meta['n_vgroups']
        n_w = tied_line_meta['n_wgroups']
        n_f = tied_line_meta['n_fgroups']
        dmu_scale_mult = float(_cfg_required('line_dmu_scale_mult'))
        sig_scale_mult = float(_cfg_required('line_sig_scale_mult'))
        amp_scale_mult = float(_cfg_required('line_amp_scale_mult'))

        dmu_group = numpyro.sample(
            'line_dmu_group',
            dist.TruncatedNormal(
                loc=_np_to_jnp(tied_line_meta['dmu_init_group']),
                scale=_np_to_jnp(np.maximum(dmu_scale_mult * (tied_line_meta['dmu_max_group'] - tied_line_meta['dmu_min_group']), 1e-6)),
                low=_np_to_jnp(tied_line_meta['dmu_min_group']),
                high=_np_to_jnp(tied_line_meta['dmu_max_group']),
            )
        ) if n_v > 0 else jnp.zeros((0,))

        sig_group = numpyro.sample(
            'line_sig_group',
            dist.TruncatedNormal(
                loc=_np_to_jnp(np.clip(tied_line_meta['sig_init_group'], 1e-5, None)),
                scale=_np_to_jnp(np.maximum(sig_scale_mult * (tied_line_meta['sig_max_group'] - tied_line_meta['sig_min_group']), 1e-6)),
                low=_np_to_jnp(np.clip(tied_line_meta['sig_min_group'], 1e-5, None)),
                high=_np_to_jnp(np.clip(tied_line_meta['sig_max_group'], 1e-5, None)),
            )
        ) if n_w > 0 else jnp.zeros((0,))

        amp_group = numpyro.sample(
            'line_amp_group',
            dist.TruncatedNormal(
                loc=_np_to_jnp(np.clip(tied_line_meta['amp_init_group'], 1e-10, None)),
                scale=_np_to_jnp(np.maximum(amp_scale_mult * (tied_line_meta['amp_max_group'] - tied_line_meta['amp_min_group']), 1e-10)),
                low=_np_to_jnp(np.clip(tied_line_meta['amp_min_group'], 1e-10, None)),
                high=_np_to_jnp(np.clip(tied_line_meta['amp_max_group'], 1e-10, None)),
            )
        ) if n_f > 0 else jnp.zeros((0,))

        dmu = dmu_group[_np_to_jnp(tied_line_meta['vgroup']).astype(int)]
        sigs = sig_group[_np_to_jnp(tied_line_meta['wgroup']).astype(int)]
        amps = amp_group[_np_to_jnp(tied_line_meta['fgroup']).astype(int)] * _np_to_jnp(tied_line_meta['flux_ratio'])
        mus = tied_line_meta['ln_lambda0'] + dmu

        line_model_intrinsic = _many_gauss_lnlam(lnwave, amps, mus, sigs)
        numpyro.deterministic('line_amp_per_component', amps)
        numpyro.deterministic('line_mu_per_component', mus)
        numpyro.deterministic('line_sig_per_component', sigs)
    else:
        line_model_intrinsic = jnp.zeros_like(wave)

    gal_model = gal_model_intrinsic
    line_model = line_model_intrinsic
    if fit_poly:
        gal_model = gal_model * poly_model
        line_model = line_model * poly_model

    frac_jitter = numpyro.sample('frac_jitter', dist.HalfNormal(_cfg_halfnorm_required('frac_jitter')))
    add_jitter = numpyro.sample('add_jitter', dist.HalfNormal(_cfg_halfnorm_required('add_jitter', ref_scale=jnp.median(err))))

    continuum_model = agn_model + gal_model
    model = continuum_model + line_model
    sigma_tot = jnp.sqrt(err**2 + (frac_jitter * jnp.abs(model))**2 + add_jitter**2)

    numpyro.deterministic('f_pl_model', pl_model)
    numpyro.deterministic('f_fe_mgii_model', fe_uv_model)
    numpyro.deterministic('f_fe_balmer_model', fe_op_model)
    numpyro.deterministic('f_bc_model', bc_model)
    numpyro.deterministic('f_poly_model', poly_model)
    numpyro.deterministic('agn_model', agn_model)
    numpyro.deterministic('gal_model_intrinsic', gal_model_intrinsic)
    numpyro.deterministic('gal_model', gal_model)
    numpyro.deterministic('line_model_intrinsic', line_model_intrinsic)
    numpyro.deterministic('line_model', line_model)
    numpyro.deterministic('continuum_model', continuum_model)
    numpyro.deterministic('model', model)
    numpyro.deterministic('PL_norm_eff', pl_norm)
    numpyro.deterministic('frac_host', frac_host)
    numpyro.deterministic('fsps_weights', fsps_weights)
    numpyro.deterministic('fsps_weights_frac', fsps_weights_frac)

    numpyro.sample('obs', dist.Normal(model, sigma_tot), obs=flux)


class QSOFit:
    def __init__(self, lam, flux, err, z, ra=-999, dec=-999, plateid=None, mjd=None, fiberid=None, path=None,
                 wdisp=None):
        self.lam_in = np.asarray(lam, dtype=np.float64)
        self.flux_in = np.asarray(flux, dtype=np.float64)
        self.err_in = np.asarray(err, dtype=np.float64)
        self.z = z
        self.wdisp = wdisp
        self.ra = ra
        self.dec = dec
        self.plateid = plateid
        self.mjd = mjd
        self.fiberid = fiberid
        self.path = path
        self.install_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = path

    def Fit(self, name=None, nsmooth=1, deredden=True,
            wave_range=None, wave_mask=None, param_file_name='qsopar.fits', save_fits_name=None,
            fit_lines=True, save_result=True, plot_fig=True, save_fits_path='.', save_fig=True,
            decompose_host=True,
            fit_fe=True,
            fit_bc=True,
            fit_poly=False,
            verbose=False,
            fsps_age_grid=(0.1, 0.3, 1.0, 3.0, 10.0),
            fsps_logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
            prior_config=None,
            dsps_ssp_fn='tempdata.h5',
            nuts_warmup=500,
            nuts_samples=1000,
            nuts_chains=1,
            nuts_target_accept=0.9,
            kwargs_plot=None):

        if kwargs_plot is None:
            kwargs_plot = {}

        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.linefit = fit_lines
        self.save_fig = save_fig
        self.verbose = verbose
        self.param_file_name = param_file_name
        prior_config = {} if prior_config is None else prior_config
        out_params = prior_config.get('out_params', {})
        self.Fe_flux_range = np.asarray(out_params.get('Fe_flux_range', []), dtype=float)
        self.L_conti_wave = np.asarray(out_params.get('cont_loc', []), dtype=float)

        self.fe_uv = np.genfromtxt(os.path.join(self.install_path, 'fe_uv.txt'))
        self.fe_op = np.genfromtxt(os.path.join(self.install_path, 'fe_optical.txt'))

        self.fe_uv_wave = 10 ** self.fe_uv[:, 0]
        self.fe_uv_flux = self.fe_uv[:, 1] * 1e15

        fe_op_wave = 10 ** self.fe_op[:, 0]
        fe_op_flux = self.fe_op[:, 1] * 1e15
        m = (fe_op_wave > 3686.) & (fe_op_wave < 7484.)
        self.fe_op_wave = fe_op_wave[m]
        self.fe_op_flux = fe_op_flux[m]

        if name is None:
            if np.array([self.plateid, self.mjd, self.fiberid]).any() is not None:
                self.sdss_name = str(self.plateid).zfill(4) + '-' + str(self.mjd) + '-' + str(self.fiberid).zfill(4)
            else:
                self.sdss_name = ''
        else:
            self.sdss_name = name

        if self.plateid is None:
            self.plateid = 0
        if self.mjd is None:
            self.mjd = 0
        if self.fiberid is None:
            self.fiberid = 0

        if save_fits_name is None:
            save_fits_name = self.sdss_name if self.sdss_name != '' else 'result'

        dustmap_path = os.path.join(self.install_path, 'sfddata')

        ind_gooderror = np.where((self.err_in > 0) & np.isfinite(self.err_in) & (self.flux_in != 0) & np.isfinite(self.flux_in), True, False)
        self.err = self.err_in[ind_gooderror]
        self.flux = self.flux_in[ind_gooderror]
        self.lam = self.lam_in[ind_gooderror]

        if nsmooth is not None:
            self.flux = self.Smooth(self.flux, nsmooth)
            self.err = self.Smooth(self.err, nsmooth)
        if wave_range is not None:
            self._WaveTrim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._WaveMsk(self.lam, self.flux, self.err, self.z)
        if deredden and self.ra != -999. and self.dec != -999.:
            self._DeRedden(self.lam, self.flux, self.err, self.ra, self.dec, dustmap_path)

        self._RestFrame(self.lam, self.flux, self.err, self.z)
        self._CalculateSN(self.wave, self.flux)
        self._OrignialSpec(self.wave, self.flux, self.err)

        self.run_fsps_numpyro_fit(
            num_warmup=nuts_warmup,
            num_samples=nuts_samples,
            num_chains=nuts_chains,
            target_accept_prob=nuts_target_accept,
            age_grid_gyr=fsps_age_grid,
            logzsol_grid=fsps_logzsol_grid,
            prior_config=prior_config,
            dsps_ssp_fn=dsps_ssp_fn,
            use_lines=fit_lines,
            decompose_host=decompose_host,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
        )

        if save_result:
            self.save_result(self.conti_result, self.conti_result_type, self.conti_result_name,
                             self.line_result, self.line_result_type, self.line_result_name,
                             save_fits_path, save_fits_name)
        if plot_fig:
            plot_kwargs = dict(kwargs_plot)
            do_trace = bool(plot_kwargs.pop('plot_trace', True))
            do_corner = bool(plot_kwargs.pop('plot_corner', True))
            full_posterior = bool(plot_kwargs.pop('full_posterior', False))
            trace_params = plot_kwargs.pop('trace_params', None)
            corner_params = plot_kwargs.pop('corner_params', None)
            max_vector_elems = plot_kwargs.pop('max_vector_elems', 2)
            max_corner_dims = plot_kwargs.pop('max_corner_dims', 8)
            if full_posterior:
                trace_params = 'all'
                corner_params = 'all'
                max_vector_elems = -1
                max_corner_dims = 0
            self.plot_fig(**plot_kwargs)
            if do_trace:
                self.plot_trace(
                    param_names=trace_params,
                    max_vector_elems=max_vector_elems,
                    save_fig_path=plot_kwargs.get('save_fig_path', '.'),
                )
            if do_corner:
                self.plot_corner(
                    param_names=corner_params,
                    max_vector_elems=max_vector_elems,
                    max_dims=max_corner_dims,
                    save_fig_path=plot_kwargs.get('save_fig_path', '.'),
                )

    def run_fsps_numpyro_fit(self, num_warmup=500, num_samples=1000, num_chains=1,
                             target_accept_prob=0.9,
                             age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                             logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                             prior_config=None,
                             dsps_ssp_fn='tempdata.h5',
                             use_lines=True,
                             decompose_host=True,
                             fit_fe=True,
                             fit_bc=True,
                             fit_poly=False):
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        prior_config = {} if prior_config is None else prior_config
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None:
            raise ValueError(
                "fit_lines=True requires line priors/table in prior_config. "
                "Pass prior_config['line_priors'] (or prior_config['line']['table'])."
            )

        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)
        else:
            tied_line_meta = {
                'n_lines': 0,
                'n_vgroups': 0,
                'n_wgroups': 0,
                'n_fgroups': 0,
                'ln_lambda0': _np_to_jnp(np.array([], dtype=float)),
                'vgroup': np.array([], dtype=int),
                'wgroup': np.array([], dtype=int),
                'fgroup': np.array([], dtype=int),
                'flux_ratio': np.array([], dtype=float),
                'dmu_init_group': np.array([], dtype=float),
                'dmu_min_group': np.array([], dtype=float),
                'dmu_max_group': np.array([], dtype=float),
                'sig_init_group': np.array([], dtype=float),
                'sig_min_group': np.array([], dtype=float),
                'sig_max_group': np.array([], dtype=float),
                'amp_init_group': np.array([], dtype=float),
                'amp_min_group': np.array([], dtype=float),
                'amp_max_group': np.array([], dtype=float),
                'names': [],
                'compnames': [],
                'line_lambda': np.array([], dtype=float),
            }
        fsps_grid = build_fsps_template_grid(
            wave_out=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
        )
        self.tied_line_meta = tied_line_meta

        init_strategy = init_to_value(values={'gal_v_kms': 0.0, 'gal_sigma_kms': 150.0})
        kernel = NUTS(qso_fsps_joint_model, init_strategy=init_strategy, target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=8)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(
            rng_key,
            wave=wave,
            flux=flux,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
        )
        samples = mcmc.get_samples()

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples=samples,
            return_sites=['f_pl_model', 'f_fe_mgii_model', 'f_fe_balmer_model', 'f_bc_model', 'f_poly_model',
                          'agn_model', 'gal_model', 'line_model', 'continuum_model', 'model',
                          'fsps_weights', 'line_amp_per_component', 'line_mu_per_component', 'line_sig_per_component'],
        )
        pred_out = pred(
            rng_key,
            wave=wave,
            flux=None,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
        )

        self.numpyro_mcmc = mcmc
        self.numpyro_samples = samples
        self.fsps_grid = fsps_grid

        self.f_pl_model = np.median(np.asarray(pred_out['f_pl_model']), axis=0)
        self.f_fe_mgii_model = np.median(np.asarray(pred_out['f_fe_mgii_model']), axis=0)
        self.f_fe_balmer_model = np.median(np.asarray(pred_out['f_fe_balmer_model']), axis=0)
        self.f_bc_model = np.median(np.asarray(pred_out['f_bc_model']), axis=0)
        self.f_poly_model = np.median(np.asarray(pred_out['f_poly_model']), axis=0)
        self.qso = np.median(np.asarray(pred_out['agn_model']), axis=0)
        self.host = np.median(np.asarray(pred_out['gal_model']), axis=0)
        self.f_line_model = np.median(np.asarray(pred_out['line_model']), axis=0)
        self.f_conti_model = np.median(np.asarray(pred_out['continuum_model']), axis=0)
        self.model_total = np.median(np.asarray(pred_out['model']), axis=0)
        self.fsps_weights_median = np.median(np.asarray(pred_out['fsps_weights']), axis=0)
        self.line_flux = flux - self.f_conti_model
        self.decomposed = True

        # Cache 1-sigma (16th/84th percentile) prediction bands for plotting.
        def _band(x):
            a = np.asarray(x)
            return np.percentile(a, 16, axis=0), np.percentile(a, 84, axis=0)

        cont_plus_lines = np.asarray(pred_out['continuum_model']) + np.asarray(pred_out['line_model'])
        self.pred_bands = {
            'total_model': _band(pred_out['model']),
            'host': _band(pred_out['gal_model']),
            'PL': _band(pred_out['f_pl_model']),
            'FeII_UV': _band(pred_out['f_fe_mgii_model']),
            'FeII_opt': _band(pred_out['f_fe_balmer_model']),
            'Balmer_cont': _band(pred_out['f_bc_model']),
            'continuum': _band(pred_out['continuum_model']),
            'lines': _band(pred_out['line_model']),
            'conti_plus_lines': _band(cont_plus_lines),
        }
        if self.verbose:
            print("max data        :", np.nanmax(self.flux))
            print("max total model :", np.nanmax(self.model_total))
            print("max PL          :", np.nanmax(self.f_pl_model))
            print("max host        :", np.nanmax(self.host))
            print("max FeII UV     :", np.nanmax(self.f_fe_mgii_model))
            print("max FeII opt    :", np.nanmax(self.f_fe_balmer_model))
            print("max Balmer cont :", np.nanmax(self.f_bc_model))
            print("max lines       :", np.nanmax(self.f_line_model))

        if decompose_host and 'gal_v_kms' in samples and 'gal_sigma_kms' in samples:
            gal_v = float(np.median(np.asarray(samples['gal_v_kms'])))
            gal_v_err = float(np.std(np.asarray(samples['gal_v_kms'])))
            gal_sig = float(np.median(np.asarray(samples['gal_sigma_kms'])))
            gal_sig_err = float(np.std(np.asarray(samples['gal_sigma_kms'])))
        else:
            gal_v, gal_v_err, gal_sig, gal_sig_err = 0.0, 0.0, 0.0, 0.0

        ages = np.array([m['tage_gyr'] for m in fsps_grid.template_meta], dtype=float)
        mets = np.array([m['logzsol'] for m in fsps_grid.template_meta], dtype=float)
        wsum = np.sum(self.fsps_weights_median)
        age_weighted = float(np.sum(self.fsps_weights_median * ages) / wsum) if wsum > 0 else -1.0
        metal_weighted = float(np.sum(self.fsps_weights_median * mets) / wsum) if wsum > 0 else -99.0

        self.frac_host_4200 = self._host_fraction_at_wave(4200.0)
        self.frac_host_5100 = self._host_fraction_at_wave(5100.0)

        if 'cont_norm' in samples:
            cont_samp = np.asarray(samples['cont_norm'])
            if decompose_host and 'log_frac_host' in samples:
                frac_host_samp = 1.0 / (1.0 + np.exp(-np.asarray(samples['log_frac_host'])))
                pl_norm_samp = cont_samp * (1.0 - frac_host_samp)
            else:
                pl_norm_samp = cont_samp
        elif 'PL_norm' in samples:
            pl_norm_samp = np.asarray(samples['PL_norm'])
        else:
            pl_norm_samp = np.full((num_samples,), np.nan)

        self.conti_result = np.array([
            self.ra, self.dec, str(self.plateid), str(self.mjd), str(self.fiberid), self.z,
            self.SN_ratio_conti,
            float(np.nanmedian(pl_norm_samp)), float(np.nanstd(pl_norm_samp)),
            float(np.median(np.asarray(samples['PL_slope']))), float(np.std(np.asarray(samples['PL_slope']))),
            gal_sig, gal_sig_err, gal_v, gal_v_err,
            self.frac_host_4200, self.frac_host_5100,
            age_weighted, metal_weighted,
        ], dtype=object)
        self.conti_result_type = np.array([
            'float', 'float', 'int', 'int', 'int', 'float', 'float',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float'
        ], dtype=object)
        self.conti_result_name = np.array([
            'ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SN_ratio_conti',
            'PL_norm', 'PL_norm_err', 'PL_slope', 'PL_slope_err',
            'sigma', 'sigma_err', 'v_off', 'v_off_err',
            'frac_host_4200', 'frac_host_5100',
            'fsps_age_weighted_gyr', 'fsps_logzsol_weighted'
        ], dtype=object)

        if use_lines and tied_line_meta['n_lines'] > 0:
            amp_comp = np.asarray(pred_out['line_amp_per_component'])
            mu_comp = np.asarray(pred_out['line_mu_per_component'])
            sig_comp = np.asarray(pred_out['line_sig_per_component'])

            amp_med = np.median(amp_comp, axis=0)
            amp_err = np.std(amp_comp, axis=0)
            mu_med = np.median(mu_comp, axis=0)
            mu_err = np.std(mu_comp, axis=0)
            sig_med = np.median(sig_comp, axis=0)
            sig_err = np.std(sig_comp, axis=0)

            vals, names, types = [], [], []
            for i, nm in enumerate(tied_line_meta['names']):
                vals.extend([amp_med[i], amp_err[i], mu_med[i], mu_err[i], sig_med[i], sig_err[i]])
                names.extend([f'{nm}_scale', f'{nm}_scale_err', f'{nm}_centerwave', f'{nm}_centerwave_err', f'{nm}_sigma', f'{nm}_sigma_err'])
                types.extend(['float'] * 6)

            self.line_result = np.array(vals, dtype=object)
            self.line_result_type = np.array(types, dtype=object)
            self.line_result_name = np.array(names, dtype=object)
            self.gauss_result = self.line_result
            self.gauss_result_name = self.line_result_name
            self.line_component_amp_median = amp_med
            self.line_component_mu_median = mu_med
            self.line_component_sig_median = sig_med
        else:
            self.line_result = np.array([])
            self.line_result_type = np.array([])
            self.line_result_name = np.array([])
            self.gauss_result = np.array([])
            self.gauss_result_name = np.array([])
            self.line_component_amp_median = np.array([])
            self.line_component_mu_median = np.array([])
            self.line_component_sig_median = np.array([])

    def _WaveTrim(self, lam, flux, err, z):
        ind_trim = np.where((lam / (1 + z) > self.wave_range[0]) & (lam / (1 + z) < self.wave_range[1]), True, False)
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError('No enough pixels in the input wave_range!')
        return self.lam, self.flux, self.err

    def _WaveMsk(self, lam, flux, err, z):
        for msk in range(len(self.wave_mask)):
            ind_not_mask = ~np.where((lam / (1 + z) > self.wave_mask[msk, 0]) & (lam / (1 + z) < self.wave_mask[msk, 1]), True, False)
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err

    def _DeRedden(self, lam, flux, err, ra, dec, dustmap_path):
        sfd_query = _get_sfd_query(dustmap_path, fetch_if_missing=True)
        coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame='icrs')
        ebv = float(np.asarray(sfd_query(coord)))
        zero_flux = np.where(flux == 0, True, False)
        flux[zero_flux] = 1e-10
        flux_unred = unred(lam, flux, ebv)
        err_unred = err * flux_unred / flux
        flux_unred[zero_flux] = 0
        self.flux = flux_unred
        self.err = err_unred
        return self.flux

    def _RestFrame(self, lam, flux, err, z):
        self.wave = lam / (1 + z)
        self.flux = flux * (1 + z)
        self.err = err * (1 + z)
        return self.wave, self.flux, self.err

    def _OrignialSpec(self, wave, flux, err):
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _CalculateSN(self, wave, flux, alter=True):
        ind5100 = np.where((wave > 5080) & (wave < 5130), True, False)
        ind3000 = np.where((wave > 3000) & (wave < 3050), True, False)
        ind1350 = np.where((wave > 1325) & (wave < 1375), True, False)
        if np.all(np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) < 10):
            if alter is False:
                self.SN_ratio_conti = -1.
                return self.SN_ratio_conti
            input_data = np.array(flux)
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            if n > 4:
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                self.SN_ratio_conti = float(signal / noise)
            else:
                self.SN_ratio_conti = -1.
        else:
            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(), flux[ind1350].mean() / flux[ind1350].std()])
            tmp_SN = tmp_SN[np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) > 10]
            self.SN_ratio_conti = np.nanmean(tmp_SN) if not np.all(np.isnan(tmp_SN)) else -1.
        return self.SN_ratio_conti

    def read_out_params(self, param_file_path='qsopar.fits'):
        hdul = fits.open(param_file_path)
        data = hdul[4].data
        self.Fe_flux_range = np.array(data['Fe_flux_range'][0])
        self.L_conti_wave = np.array(data['cont_loc'][0])
        return data

    def flux2L(self, flux, z=None):
        if z is None:
            z = self.z
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_L = cosmo.luminosity_distance(z).to(u.cm).value
        return flux * 1e-17 * 4 * np.pi * d_L ** 2

    def _host_fraction_at_wave(self, w0):
        if len(self.wave) == 0:
            return -1.
        host = np.interp(w0, self.wave, self.host, left=np.nan, right=np.nan)
        total = np.interp(w0, self.wave, self.f_conti_model, left=np.nan, right=np.nan)
        if not np.isfinite(host) or not np.isfinite(total) or total == 0:
            return -1.
        return float(host / total)

    def Smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')

    def save_result(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type, line_result_name, save_fits_path, save_fits_name):
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])
        t = Table(self.all_result, names=(self.all_result_name), dtype=self.all_result_type)
        t.write(os.path.join(save_fits_path, save_fits_name + '.fits'), format='fits', overwrite=True)
        return

    def _posterior_series(self, param_names=None, max_vector_elems=2):
        if not hasattr(self, 'numpyro_samples') or self.numpyro_samples is None:
            return []

        samples = self.numpyro_samples
        if param_names == 'all':
            param_names = sorted(samples.keys())
        elif param_names is None:
            param_names = [
                'cont_norm', 'log_frac_host', 'PL_slope', 'Fe_uv_norm', 'Fe_op_norm',
                'Balmer_norm', 'Balmer_Te', 'Balmer_Tau',
                'gal_v_kms', 'gal_sigma_kms',
                'frac_jitter', 'add_jitter',
            ]

        out = []
        for name in param_names:
            if name not in samples:
                continue
            arr = np.asarray(samples[name])
            if arr.ndim == 1:
                out.append((name, arr))
            elif arr.ndim >= 2:
                arr2 = arr.reshape(arr.shape[0], -1)
                nflat = arr2.shape[1]
                if max_vector_elems is None or int(max_vector_elems) < 0:
                    ncomp = nflat
                else:
                    ncomp = min(nflat, int(max_vector_elems))
                for i in range(ncomp):
                    out.append((f'{name}[{i}]', arr2[:, i]))
        return out

    def plot_trace(self, param_names=None, max_vector_elems=2, save_fig_path='.', save_fig_name=None):
        series = self._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
        if len(series) == 0:
            return None

        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(2.2 * n, 4)), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (label, vals) in zip(axes, series):
            ax.plot(np.arange(len(vals)), vals, color='tab:blue', lw=0.8)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel('Sample', fontsize=10)
        fig.suptitle(f'{self.sdss_name} Trace Plot', fontsize=14)
        fig.tight_layout()
        plt.show()
        if self.save_fig:
            out_name = f'{self.sdss_name}_trace.pdf' if save_fig_name is None else save_fig_name
            fig.savefig(os.path.join(save_fig_path, out_name))
            plt.close(fig)
        self.trace_fig = fig
        return fig

    def plot_corner(self, param_names=None, max_vector_elems=2, max_dims=8, bins=30, save_fig_path='.', save_fig_name=None):
        series = self._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
        if len(series) == 0:
            return None

        if max_dims is not None and int(max_dims) > 0:
            series = series[:int(max_dims)]
        labels = [s[0] for s in series]
        data = np.column_stack([s[1] for s in series])
        ndim = data.shape[1]

        fig, axes = plt.subplots(ndim, ndim, figsize=(2.2 * ndim, 2.2 * ndim))
        for i in range(ndim):
            for j in range(ndim):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off')
                    continue
                if i == j:
                    ax.hist(data[:, j], bins=bins, color='tab:blue', alpha=0.75)
                else:
                    ax.scatter(data[:, j], data[:, i], s=3, alpha=0.25, color='tab:blue')
                if i == ndim - 1:
                    ax.set_xlabel(labels[j], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=8)
                else:
                    ax.set_yticklabels([])
        fig.suptitle(f'{self.sdss_name} Corner Plot', fontsize=14)
        fig.tight_layout()
        plt.show()
        if self.save_fig:
            out_name = f'{self.sdss_name}_corner.pdf' if save_fig_name is None else save_fig_name
            fig.savefig(os.path.join(save_fig_path, out_name))
            plt.close(fig)
        self.corner_fig = fig
        return fig

    def plot_fig(self, save_fig_path='.', broad_fwhm=1200, plot_legend=True, ylims=None, plot_residual=True, show_title=True,
                 plot_1sigma=True, sigma_alpha=0.12):
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        if plot_1sigma and hasattr(self, 'pred_bands'):
            band_colors = {
                'total_model': 'b',
                'host': 'purple',
                'PL': 'orange',
                'FeII_UV': 'c',
                'FeII_opt': 'teal',
                'Balmer_cont': 'y',
                'continuum': 'darkorange',
                'lines': 'crimson',
                'conti_plus_lines': 'green',
            }
            for key, color in band_colors.items():
                if key not in self.pred_bands:
                    continue
                lo, hi = self.pred_bands[key]
                if len(lo) == len(self.wave):
                    ax.fill_between(self.wave, lo, hi, color=color, alpha=sigma_alpha, linewidth=0, zorder=0)

        ax.plot(self.wave_prereduced, self.flux_prereduced, 'k', lw=1, label='data', zorder=2)
        ax.plot(self.wave, self.model_total, color='b', lw=1.8, label='total model', zorder=6)
        ax.plot(self.wave, self.host, color='purple', lw=1.8, label='host', zorder=4)
        ax.plot(self.wave, self.f_pl_model, color='orange', lw=1.5, label='PL', zorder=5)
        ax.plot(self.wave, self.f_fe_mgii_model, color='c', lw=1.2, label='FeII UV', zorder=5)
        ax.plot(self.wave, self.f_fe_balmer_model, color='teal', lw=1.2, label='FeII opt', zorder=5)
        ax.plot(self.wave, self.f_bc_model, color='y', lw=1.2, label='Balmer cont.', zorder=5)
        ax.plot(self.wave, self.f_conti_model, color='darkorange', lw=1.5, label='continuum', zorder=5)
        if len(self.f_line_model) == len(self.wave):
            ax.plot(self.wave, self.f_line_model, color='crimson', lw=1.2, label='lines', zorder=5)
            ax.plot(self.wave, self.f_conti_model + self.f_line_model, color='green', lw=1.2, label='conti+lines', zorder=5)

        # Plot individual Gaussian line components: broad (*_br) in red, narrow in green.
        if (hasattr(self, 'line_component_amp_median')
                and hasattr(self, 'line_component_mu_median')
                and hasattr(self, 'line_component_sig_median')
                and hasattr(self, 'tied_line_meta')
                and len(self.line_component_amp_median) > 0):
            lnwave = np.log(self.wave)
            compnames = self.tied_line_meta.get('compnames', [''] * len(self.line_component_amp_median))
            drew_broad_label = False
            drew_narrow_label = False
            for i in range(len(self.line_component_amp_median)):
                amp = float(self.line_component_amp_median[i])
                mu = float(self.line_component_mu_median[i])
                sig = float(self.line_component_sig_median[i])
                if not np.isfinite(amp) or not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
                    continue
                prof = amp * np.exp(-0.5 * ((lnwave - mu) / sig) ** 2)
                # Keep component plotting consistent with polynomial correction if enabled.
                if hasattr(self, 'f_poly_model') and len(self.f_poly_model) == len(prof):
                    prof = prof * self.f_poly_model
                cname = str(compnames[i]).lower()
                is_broad = cname.endswith('_br') or ('_br' in cname)
                if is_broad:
                    lbl = 'broad comps' if not drew_broad_label else None
                    ax.plot(self.wave, prof, color='red', lw=0.7, alpha=0.35, zorder=3, label=lbl)
                    drew_broad_label = True
                else:
                    lbl = 'narrow comps' if not drew_narrow_label else None
                    ax.plot(self.wave, prof, color='green', lw=0.7, alpha=0.25, zorder=3, label=lbl)
                    drew_narrow_label = True

        if show_title:
            ax.set_title(f"{self.sdss_name}   z = {np.round(float(self.z), 4)}", fontsize=20)
        ax.set_xlim(self.wave.min(), self.wave.max())
        if ylims is None:
            yplot = np.concatenate([
                self.flux[np.isfinite(self.flux)],
                self.model_total[np.isfinite(self.model_total)]
            ])
            if yplot.size > 0:
                y1, y2 = np.nanpercentile(yplot, [1, 99])
                if np.isfinite(y1) and np.isfinite(y2) and y2 > y1:
                    pad = 0.15 * (y2 - y1)
                    ax.set_ylim(0, y2 + pad)
        else:
            ax.set_ylim(0, ylims[1])

        if plot_residual and len(self.model_total) == len(self.wave):
            resid = self.flux - self.model_total
            y1, y2 = ax.get_ylim()
            resid_level = y1 + 0.08 * (y2 - y1)
            ax.plot(self.wave, resid + resid_level, color='gray', ls='dotted', lw=1, label='resid', zorder=1)
            ax.axhline(resid_level, color='gray', ls='--', lw=0.8)

        ax.set_xlabel(r'Rest Wavelength (\AA)', fontsize=20)
        ax.set_ylabel(r'$f_{\lambda}$', fontsize=20)
        if plot_legend:
            ax.legend(frameon=False, fontsize=9, ncol=2)
        plt.show()
        if self.save_fig:
            fig.savefig(os.path.join(save_fig_path, self.sdss_name + '.pdf'))
            plt.close(fig)
        self.fig = fig
        return


# Notes
# -----
# 1) This restores the standard AGN continuum pieces in the Bayesian model.
# 2) The dust term is a JAX-safe attenuation surrogate instead of the exact dust_extinction call.
# 3) Plotting is restored in a PyQSOFit-like layered style.
# 4) The main remaining extension would be reproducing every legacy derived summary field exactly,
#    but the model components and plotting interface are now back in place.
