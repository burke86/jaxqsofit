import numpy as np

from jaxqsofit.defaults import build_default_bal_components, build_default_prior_config


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
    assert cfg["host_redshift_prior"]["enabled"] is False
    assert cfg["host_redshift_prior"]["z_mid"] == 1.0
    assert cfg["host_redshift_prior"]["width"] == 0.2
    assert cfg["host_redshift_prior"]["lowz_loc_offset"] == 0.0
    assert cfg["host_redshift_prior"]["highz_loc_offset"] == -8.0
    assert cfg["host_redshift_prior"]["lowz_scale_mult"] == 1.0
    assert cfg["host_redshift_prior"]["highz_scale_mult"] == 0.05
    assert cfg["host_redshift_prior"]["lowz_df"] == 3.0
    assert cfg["host_redshift_prior"]["highz_df"] == 20.0
    assert cfg["log_Fe_uv_FWHM"]["dist"] == "LogNormal"
    assert cfg["Fe_uv_shift"]["dist"] == "Normal"
    assert cfg["frac_jitter"]["dist"] == "HalfNormal"
    assert cfg["add_jitter"]["dist"] == "HalfNormal"


def test_build_default_bal_components_exposes_common_bal_lines():
    comps = build_default_bal_components(np.array([1.0, 2.0, 3.0], dtype=float))

    names = [comp.name for comp in comps]
    assert names == ["bal_nv", "bal_siiv", "bal_civ", "bal_ciii", "bal_fe1", "bal_fe2", "bal_mgii"]
    depth_cfg = comps[2].parameter_priors["depth"]
    assert depth_cfg["dist"] == "HalfNormal"
    assert np.isclose(depth_cfg["scale"], 8.0 * 0.05 * 2.0)
    center_cfg = comps[2].parameter_priors["center"]
    assert center_cfg["dist"] == "TruncatedNormal"
    assert center_cfg["loc"] == 1500.0
    assert center_cfg["low"] == 1400.0
    assert center_cfg["high"] == 1549.0
    shape_cfg = comps[2].parameter_priors["shape_power"]
    assert shape_cfg["dist"] == "TruncatedNormal"
    assert shape_cfg["loc"] == 2.0
    assert shape_cfg["low"] == 2.0
    assert shape_cfg["high"] == 12.0


def test_default_line_table_contains_expanded_uv_complexes():
    cfg = build_default_prior_config(np.array([1.0, 2.0, 3.0], dtype=float))
    rows = cfg["line"]["table"]
    by_name = {row["linename"]: row for row in rows}

    expected_names = {
        "SiIII1892",
        "AlIII1857",
        "SiII1816",
        "NIII1750",
        "NIV1718",
        "CIV_br",
        "HeII1640",
        "OIII1663",
        "HeII1640_br",
        "OIII1663_br",
        "SiIV_OIV1",
        "SiIV_OIV2",
        "CII1335",
        "OI1304",
        "Lya_br",
        "NV1240",
    }
    assert expected_names.issubset(by_name)
    assert "CIII_br" not in by_name
    assert by_name["CIV_br"]["ngauss"] == 3
    assert by_name["Lya_br"]["ngauss"] == 3
