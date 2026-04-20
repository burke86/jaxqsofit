import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace

import jaxqsofit.model as modelmod
from jaxqsofit.defaults import build_default_prior_config
from jaxqsofit.custom_components import make_custom_component
from jaxqsofit.model import (
    _host_redshift_prior_params,
    _extract_line_table_from_prior_config,
    _luminosity_distance_cm_jax,
    build_tied_line_meta_from_linelist,
    build_fsps_template_grid,
    clear_fsps_template_grid_cache,
    fsps_template_grid_cache_info,
    qso_fsps_joint_model,
)


class _FakeSSPData:
    ssp_lgmet = np.asarray([-1.0, 0.0], dtype=float)
    ssp_lg_age_gyr = np.asarray([-1.0, 0.0], dtype=float)
    ssp_wave = np.linspace(1000.0, 9000.0, 8)
    ssp_flux = np.arange(2 * 2 * 8, dtype=float).reshape(2, 2, 8) + 1.0


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


def test_build_fsps_template_grid_reuses_cached_grid(monkeypatch, tmp_path):
    calls = {"load": 0}

    def _fake_load_ssp_templates(fn):
        calls["load"] += 1
        return _FakeSSPData()

    monkeypatch.setattr(modelmod, "load_ssp_templates", _fake_load_ssp_templates)
    clear_fsps_template_grid_cache()

    dsps_path = tmp_path / "fake_ssp.h5"
    dsps_path.write_text("first")
    wave = np.linspace(2000.0, 7000.0, 16)
    kwargs = {
        "wave_out": wave,
        "age_grid_gyr": (0.1, 1.0),
        "logzsol_grid": (-1.0, 0.0),
        "dsps_ssp_fn": str(dsps_path),
    }

    grid1 = build_fsps_template_grid(**kwargs)
    grid1.templates[:] = -123.0
    grid2 = build_fsps_template_grid(**kwargs)

    assert calls["load"] == 1
    assert fsps_template_grid_cache_info() == {"size": 1}
    assert not np.all(grid2.templates == -123.0)
    assert np.array_equal(grid2.wave, wave)

    clear_fsps_template_grid_cache()


def test_build_fsps_template_grid_cache_invalidates_when_file_changes(monkeypatch, tmp_path):
    calls = {"load": 0}

    def _fake_load_ssp_templates(fn):
        calls["load"] += 1
        return _FakeSSPData()

    monkeypatch.setattr(modelmod, "load_ssp_templates", _fake_load_ssp_templates)
    clear_fsps_template_grid_cache()

    dsps_path = tmp_path / "fake_ssp.h5"
    dsps_path.write_text("first")
    wave = np.linspace(2000.0, 7000.0, 16)
    kwargs = {
        "wave_out": wave,
        "age_grid_gyr": (0.1, 1.0),
        "logzsol_grid": (-1.0, 0.0),
        "dsps_ssp_fn": str(dsps_path),
    }

    build_fsps_template_grid(**kwargs)
    dsps_path.write_text("second version")
    build_fsps_template_grid(**kwargs)

    assert calls["load"] == 2
    assert fsps_template_grid_cache_info() == {"size": 2}

    clear_fsps_template_grid_cache()


def test_qso_fsps_joint_model_reports_log_lambda_llambda_requested_continuum_luminosities():
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

    for wave_label in ("1350", "2500", "3000", "5100"):
        site_name = f"log_lambda_Llambda_{wave_label}_agn"
        assert site_name in tr
        assert np.isfinite(float(tr[site_name]["value"]))


def test_luminosity_distance_cm_jax_is_finite_and_vectorizable():
    z = jnp.asarray([0.1, 1.0, 2.0])
    d_l = _luminosity_distance_cm_jax(z)

    assert d_l.shape == (3,)
    assert np.all(np.isfinite(np.asarray(d_l)))
    assert np.all(np.asarray(d_l) > 0.0)
    assert np.all(np.diff(np.asarray(d_l)) > 0.0)


def test_qso_fsps_joint_model_custom_component_returns_jax_array():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    comp = make_custom_component(
        name="const_term",
        parameter_priors={"c0": {"dist": "Normal", "loc": 0.0, "scale": 1.0}},
        evaluate=lambda wave, params, metadata: jnp.zeros_like(wave) + params["c0"],
    )

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
        "custom_const_term_c0": np.array(0.5),
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
        z_qso=jnp.asarray(1.0),
        custom_components=[comp],
    )

    value = tr["custom_const_term_model"]["value"]
    assert isinstance(value, jax.Array)
    assert np.all(np.isfinite(np.asarray(value)))


def test_host_redshift_prior_params_shift_negative_at_high_z():
    cfg = {
        "host_redshift_prior": {
            "enabled": True,
            "z_mid": 1.0,
            "width": 0.2,
            "lowz_loc_offset": 0.0,
            "highz_loc_offset": -8.0,
            "lowz_scale_mult": 1.0,
            "highz_scale_mult": 0.05,
            "lowz_df": 3.0,
            "highz_df": 20.0,
        }
    }
    w_low, offset_low, scale_low, df_low = _host_redshift_prior_params(cfg, 0.2)
    w_mid, offset_mid, scale_mid, df_mid = _host_redshift_prior_params(cfg, 1.0)
    w_high, offset_high, scale_high, df_high = _host_redshift_prior_params(cfg, 2.0)

    assert float(w_low) < 0.1
    assert np.isclose(float(offset_low), 0.0, atol=0.15)
    assert np.isclose(float(scale_low), 1.0, atol=0.1)
    assert np.isclose(float(df_low), 3.0, atol=1.0)
    assert np.isclose(float(w_mid), 0.5, atol=1e-6)
    assert np.isclose(float(offset_mid), -4.0, atol=1e-6)
    assert np.isclose(float(scale_mid), 0.525, atol=1e-6)
    assert np.isclose(float(df_mid), 11.5, atol=1e-6)
    assert float(w_high) > 0.99
    assert np.isclose(float(offset_high), -7.95, atol=0.05)
    assert np.isclose(float(scale_high), 0.056, atol=0.02)
    assert np.isclose(float(df_high), 19.9, atol=0.2)


def test_host_redshift_prior_params_disable_restores_zero_offset():
    w, offset, scale_mult, df_eff = _host_redshift_prior_params({"host_redshift_prior": {"enabled": False}}, 2.0)
    assert float(w) == 0.0
    assert float(offset) == 0.0
    assert float(scale_mult) == 1.0
    assert df_eff is None


def test_host_redshift_prior_penalizes_same_host_more_at_high_z():
    cfg = build_default_prior_config(np.array([1.0, 2.0, 3.0], dtype=float))
    cfg["host_redshift_prior"]["enabled"] = True
    base = cfg["log_frac_host"]
    _, offset_low, scale_low, df_low = _host_redshift_prior_params(cfg, 0.2)
    _, offset_high, scale_high, df_high = _host_redshift_prior_params(cfg, 2.0)

    x = jnp.asarray(1.5)
    logp_low = dist.StudentT(df=float(df_low), loc=float(base["loc"] + offset_low), scale=float(base["scale"] * scale_low)).log_prob(x)
    logp_high = dist.StudentT(df=float(df_high), loc=float(base["loc"] + offset_high), scale=float(base["scale"] * scale_high)).log_prob(x)

    assert float(logp_high) < float(logp_low)


def test_qso_fsps_joint_model_reports_host_redshift_prior_diagnostics():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)
    cfg["host_redshift_prior"]["enabled"] = True

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(-1.0),
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
        z_qso=2.0,
    )

    assert "host_redshift_prior_weight" in tr
    assert "host_redshift_prior_loc_eff" in tr
    assert "host_redshift_prior_scale_eff" in tr
    assert "host_redshift_prior_df_eff" in tr
    assert float(tr["host_redshift_prior_weight"]["value"]) > 0.99
    assert np.isclose(float(tr["host_redshift_prior_loc_eff"]["value"]), -7.95, atol=0.05)
    assert np.isclose(float(tr["host_redshift_prior_scale_eff"]["value"]), 0.11, atol=0.05)
    assert np.isclose(float(tr["host_redshift_prior_df_eff"]["value"]), 19.9, atol=0.2)
