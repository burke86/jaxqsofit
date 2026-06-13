import numpy as np

from jaxqsofit.defaults import (
    DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS,
    DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS,
    build_default_bal_components,
    build_default_prior_config,
)


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
        'host_sfh_model',
        'line_dmu_scale_mult',
        'line_sig_scale_mult',
        'line_amp_scale_mult',
        'line',
        'student_t_df',
    ]
    for k in required:
        assert k in cfg
    assert cfg["host_sfh_model"] == "delayed"
    assert cfg["log_stellar_mass"] == {
        "dist": "TruncatedNormal",
        "loc": 9.0,
        "scale": 0.75,
        "low": 7.0,
        "high": 12.0,
    }
    assert cfg["log_host_aperture_scale"] == {"dist": "Normal", "loc": 0.0, "scale": 0.5}


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
        "CIII_br",
        "CIII_na",
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
    assert "CIII_br" in by_name
    assert by_name["CIII_br"]["ngauss"] == 2
    assert by_name["CIV_br"]["ngauss"] == 3
    assert by_name["Lya_br"]["ngauss"] == 3


def test_optional_line_tables_do_not_duplicate_hei7065():
    cfg = build_default_prior_config(
        np.array([1.0, 2.0, 3.0], dtype=float),
        include_elg_narrow_lines=True,
        include_high_ionization_lines=True,
    )
    rows = cfg["line"]["table"]
    hei7065 = [row for row in rows if row["linename"] == "HeI7065"]

    assert len(hei7065) == 1
    assert np.isclose(hei7065[0]["lambda"], 7067.17)


def test_optional_fixed_doublet_ratios_are_physical():
    elg_by_name = {row["linename"]: row for row in DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS}
    high_ion_by_name = {row["linename"]: row for row in DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS}

    assert np.isclose(
        elg_by_name["OIII5007"]["fvalue"] * elg_by_name["OIII5007"]["lambda"]
        / (elg_by_name["OIII4959"]["fvalue"] * elg_by_name["OIII4959"]["lambda"]),
        2.98,
    )
    assert np.isclose(
        elg_by_name["OI6300"]["fvalue"] * elg_by_name["OI6300"]["lambda"]
        / (elg_by_name["OI6363"]["fvalue"] * elg_by_name["OI6363"]["lambda"]),
        3.05,
    )
    assert np.isclose(
        elg_by_name["NII6583"]["fvalue"] * elg_by_name["NII6583"]["lambda"]
        / (elg_by_name["NII6548"]["fvalue"] * elg_by_name["NII6548"]["lambda"]),
        3.0,
    )
    assert np.isclose(
        high_ion_by_name["NeV3426_hi"]["fvalue"] * high_ion_by_name["NeV3426_hi"]["lambda"]
        / (high_ion_by_name["NeV3346"]["fvalue"] * high_ion_by_name["NeV3346"]["lambda"]),
        2.7,
    )


def test_combined_optional_config_preserves_oi_doublet_ratio():
    cfg = build_default_prior_config(
        np.array([1.0, 2.0, 3.0], dtype=float),
        include_elg_narrow_lines=True,
    )
    by_name = {row["linename"]: row for row in cfg["line"]["table"]}

    assert np.isclose(
        by_name["OI6300"]["fvalue"] * by_name["OI6300"]["lambda"]
        / (by_name["OI6363"]["fvalue"] * by_name["OI6363"]["lambda"]),
        3.05,
    )
