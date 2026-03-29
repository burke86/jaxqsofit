import numpy as np
import jax
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.util import log_density

from jaxqsofit.defaults import build_default_prior_config
from jaxqsofit.model import (
    _extract_line_table_from_prior_config,
    _host_luminosity_penalty_terms,
    build_tied_line_meta_from_linelist,
    qso_fsps_joint_model,
)


def test_extract_line_table_from_prior_config_layouts():
    table = [{'lambda': 5008.24, 'linename': 'OIII5007', 'compname': 'Hb', 'ngauss': 1, 'inisca': 1.0, 'minsca': 0.0, 'maxsca': 1e3, 'inisig': 1e-3, 'minsig': 1e-4, 'maxsig': 1e-2, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 1.0}]

    cfg1 = {'line_priors': table}
    cfg2 = {'line_table': table}
    cfg3 = {'line': {'table': table}}
    cfg4 = {'line': {'priors': table}}

    assert _extract_line_table_from_prior_config(cfg1) is table
    assert _extract_line_table_from_prior_config(cfg2) is table
    assert _extract_line_table_from_prior_config(cfg3) is table
    assert _extract_line_table_from_prior_config(cfg4) is table


def test_build_tied_line_meta_from_linelist_minimal():
    line_table = [
        {
            'lambda': 5008.24,
            'linename': 'OIII5007',
            'compname': 'Hb',
            'ngauss': 1,
            'inisca': 1.0,
            'minsca': 0.0,
            'maxsca': 1e3,
            'inisig': 1e-3,
            'minsig': 1e-4,
            'maxsig': 1e-2,
            'voff': 0.01,
            'vindex': 1,
            'windex': 1,
            'findex': 1,
            'fvalue': 1.0,
        },
        {
            'lambda': 4960.30,
            'linename': 'OIII4959',
            'compname': 'Hb',
            'ngauss': 1,
            'inisca': 0.3,
            'minsca': 0.0,
            'maxsca': 1e3,
            'inisig': 1e-3,
            'minsig': 1e-4,
            'maxsig': 1e-2,
            'voff': 0.01,
            'vindex': 1,
            'windex': 1,
            'findex': 1,
            'fvalue': 0.33,
        },
    ]
    wave = np.linspace(4800.0, 5100.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta['n_lines'] == 2
    assert meta['n_vgroups'] >= 1
    assert meta['n_wgroups'] >= 1
    assert meta['n_fgroups'] >= 1
    assert len(meta['names']) == 2
    assert np.all(np.isfinite(meta['line_lambda']))


def test_host_luminosity_penalty_terms_transition_with_luminosity():
    cfg = {
        "log_lambda_Llambda_mid": 45.2,
        "width_dex": 0.3,
        "max_logit_shift": 100.0,
    }
    weight_low, penalty_low = _host_luminosity_penalty_terms(0.0, 44.2, cfg)
    weight_mid, penalty_mid = _host_luminosity_penalty_terms(0.0, 45.2, cfg)
    weight_high, penalty_high = _host_luminosity_penalty_terms(0.0, 46.2, cfg)

    assert float(weight_low) < 0.1
    assert np.isclose(float(weight_mid), 0.5, atol=1e-6)
    assert float(weight_high) > 0.9
    assert float(penalty_high) < float(penalty_mid) < float(penalty_low) <= 0.0


def test_qso_fsps_joint_model_host_penalty_enabled_by_default():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(1.0),
        "PL_norm": np.array(5.0e6),
        "PL_slope": np.array(0.0),
        "tau_host": np.array(1.0),
        "fsps_weights_raw": np.array([0.0]),
        "gal_v_kms": np.array(0.0),
        "gal_sigma_kms": np.array(100.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=1.0,
    )

    assert np.isfinite(float(tr["log_lambda_Llambda_2500_agn"]["value"]))
    assert float(tr["host_luminosity_penalty_weight"]["value"]) > 0.0
    assert float(tr["host_luminosity_penalty_value"]["value"]) < 0.0


def test_qso_fsps_joint_model_host_penalty_increases_at_high_luminosity():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)
    cfg["host_luminosity_penalty"] = {
        "enabled": True,
        "wave": 2500.0,
        "log_lambda_Llambda_mid": 45.2,
        "width_dex": 0.3,
        "max_logit_shift": 100.0,
    }

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    base_params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(1.0),
        "PL_slope": np.array(0.0),
        "tau_host": np.array(1.0),
        "fsps_weights_raw": np.array([0.0]),
        "gal_v_kms": np.array(0.0),
        "gal_sigma_kms": np.array(100.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }

    params = dict(base_params, PL_norm=np.array(5.0e3))

    def _logp(z_qso):
        lp, _ = log_density(
            qso_fsps_joint_model,
            (
                wave,
                flux,
                err,
                {},
                {"n_lines": 0},
                _Grid(),
                np.array([2000.0, 6000.0]),
                np.zeros(2),
                np.array([2000.0, 6000.0]),
                np.zeros(2),
            ),
            {
                "use_lines": False,
                "prior_config": cfg,
                "decompose_host": True,
                "fit_pl": True,
                "fit_fe": False,
                "fit_bc": False,
                "fit_poly": False,
                "fit_reddening": False,
                "z_qso": z_qso,
            },
            params,
        )
        return float(lp)

    tr_low = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=0.01,
    )
    tr_high = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=5.0,
    )

    assert float(tr_low["host_luminosity_penalty_weight"]["value"]) < 0.1
    assert np.isfinite(float(tr_high["log_lambda_Llambda_2500_agn"]["value"]))
    assert float(tr_high["host_luminosity_penalty_weight"]["value"]) > 0.5
    assert float(tr_high["host_luminosity_penalty_value"]["value"]) < float(tr_low["host_luminosity_penalty_value"]["value"])
    assert _logp(5.0) < _logp(0.01)
