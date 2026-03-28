import numpy as np

from jaxqsofit.defaults import build_default_prior_config


def test_build_default_prior_config_has_expected_keys():
    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    cfg = build_default_prior_config(flux)

    required = [
        'log_cont_norm',
        'PL_norm',
        'PL_slope',
        'PL_pivot',
        'reddening_ebv',
        'log_frac_host',
        'host_luminosity_penalty',
        'tau_host',
        'raw_w',
        'line_dmu_scale_mult',
        'line_sig_scale_mult',
        'line_amp_scale_mult',
        'line',
        'student_t_df',
    ]
    for k in required:
        assert k in cfg


def test_build_default_prior_config_scales_with_flux_median():
    flux = np.array([10.0, 20.0, 30.0], dtype=float)
    cfg = build_default_prior_config(flux)

    expected = np.log(np.median(np.abs(flux)))
    got = float(cfg['log_cont_norm']['loc'])
    assert np.isfinite(got)
    assert np.isclose(got, expected)
    assert cfg['log_cont_norm']['dist'] == 'LogNormal'
    assert cfg['PL_slope']['dist'] == 'Normal'
    assert cfg['tau_host']['dist'] == 'HalfNormal'


def test_build_default_prior_config_accepts_manual_pl_pivot():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    cfg = build_default_prior_config(flux, pl_pivot=3000.0)
    assert cfg["PL_pivot"] == 3000.0


def test_build_default_prior_config_uses_explicit_dist_fields():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    cfg = build_default_prior_config(flux)

    assert cfg["log_frac_host"]["dist"] == "StudentT"
    assert cfg["log_Fe_uv_FWHM"]["dist"] == "LogNormal"
    assert cfg["Fe_uv_shift"]["dist"] == "Normal"
    assert cfg["frac_jitter"]["dist"] == "HalfNormal"
    assert cfg["add_jitter"]["dist"] == "HalfNormal"


def test_build_default_prior_config_includes_host_luminosity_penalty_defaults():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    cfg = build_default_prior_config(flux)

    penalty = cfg["host_luminosity_penalty"]
    assert penalty["enabled"] is True
    assert penalty["wave"] == 2500.0
    assert penalty["log_lambda_Llambda_mid"] == 45.2
    assert penalty["width_dex"] == 0.3
    assert penalty["max_logit_shift"] == 4.0
