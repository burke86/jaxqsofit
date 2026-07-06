import numpy as np
import numpyro.distributions as dist
import pytest

from jaxqsofit.config import (
    BALConfig,
    ContinuumConfig,
    ContinuumPriorConfig,
    FeIIPriorConfig,
    FitConfig,
    HostConfig,
    HostPriorConfig,
    LinePriorConfig,
    Observation,
    PriorConfig,
    SpectroscopyData,
)
from jaxqsofit.defaults import (
    DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS,
    DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS,
    _build_default_prior_config as build_default_prior_config,
    build_default_bal_components,
)


def test_prior_config_object_exposes_model_mapping():
    prior = PriorConfig(
        continuum=ContinuumPriorConfig(power_law_pivot=3000.0, polynomial_pivot=2800.0),
        host=HostPriorConfig(redshift_weight_enabled=False),
        lines=LinePriorConfig(dmu_scale_mult=0.2, sig_scale_mult=0.3, amp_scale_mult=0.4),
        feii=FeIIPriorConfig(uv_fwhm=dist.Normal(loc=np.log(1000.0), scale=0.2)),
    )
    prior.powerlaw.slope = dist.Normal(loc=-1.5, scale=0.3)
    mapping = prior.to_mapping()

    assert mapping["PL_pivot"] == 3000.0
    assert mapping["poly_pivot"] == 2800.0
    assert mapping["host_redshift_prior"]["enabled"] is False
    assert mapping["line_dmu_scale_mult"] == 0.2
    assert mapping["log_Fe_uv_FWHM"]["scale"] == 0.2
    assert mapping["PL_slope"] == {"dist": "Normal", "loc": -1.5, "scale": 0.3}


def test_fit_config_to_dict_serializes_numpyro_priors():
    prior = PriorConfig()
    prior.powerlaw.slope = dist.Normal(loc=-1.5, scale=0.3)
    cfg = FitConfig(
        observation=Observation(redshift=0.1),
        spectroscopy=SpectroscopyData(wave_obs=[4000.0, 5000.0], fluxes=[1.0, 1.1]),
        prior_config=prior,
    )

    data = cfg.to_dict()

    assert data["prior_config"]["continuum"]["powerlaw"]["slope"] == {
        "dist": "Normal",
        "loc": -1.5,
        "scale": 0.3,
    }


def test_continuum_config_broadening_convolution_default_and_validation():
    assert ContinuumConfig().broadening_convolution == "fft"
    assert ContinuumConfig(broadening_convolution="direct").broadening_convolution == "direct"
    with pytest.raises(ValueError, match="broadening_convolution"):
        ContinuumConfig(broadening_convolution="scipy")


def test_prior_config_from_spectrum_exposes_semantic_prior_sections():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)

    prior = PriorConfig.from_spectrum(
        flux=flux,
        redshift=1.0,
        include_high_ionization_lines=False,
        include_elg_narrow_lines=False,
    )
    prior.powerlaw.slope = dist.TruncatedNormal(loc=-1.5, scale=0.3, low=-3.5, high=0.5)
    prior.fe.uv_norm = dist.LogNormal(np.log(max(1.0e-3 * np.median(np.abs(flux)), 1.0e-10)), 0.04)
    prior.fe.op_over_uv = dist.Normal(loc=0.0, scale=0.4)
    prior.lines.dmu_scale_mult = 0.25
    prior.lines.sig_scale_mult = 0.25
    prior.lines.amp_scale_mult = 0.20
    prior.host.stellar_mass = dist.TruncatedNormal(loc=10.6, scale=0.4, low=9.5, high=12.0)
    prior.host.aperture_scale = dist.Normal(loc=0.0, scale=0.5)
    prior.host.sfh_age_gyr = dist.Normal(loc=np.log(7.0), scale=0.3)
    prior.host.sfh_tau_over_age = dist.Normal(loc=np.log(0.25), scale=0.3)
    prior.host.metallicity = dist.Normal(loc=0.0, scale=0.2)
    prior.host.metallicity_scatter = dist.Normal(loc=np.log(0.1), scale=0.3)
    prior.host.template_age_prior = {"type": "prefer_old", "pivot_gyr": 1.0, "strength": 2.0}
    prior.host.sfh_model = "delayed"

    mapping = prior.to_mapping()

    assert mapping["PL_slope"] == {"dist": "TruncatedNormal", "loc": -1.5, "scale": 0.3, "low": -3.5, "high": 0.5}
    assert mapping["log_Fe_uv_norm"] == {
        "dist": "LogNormal",
        "loc": np.log(max(1.0e-3 * np.median(np.abs(flux)), 1.0e-10)),
        "scale": 0.04,
    }
    assert mapping["log_Fe_op_over_uv"] == {"dist": "Normal", "loc": 0.0, "scale": 0.4}
    assert mapping["line_dmu_scale_mult"] == 0.25
    assert mapping["line_sig_scale_mult"] == 0.25
    assert mapping["line_amp_scale_mult"] == 0.20
    assert mapping["log_stellar_mass"] == {"dist": "TruncatedNormal", "loc": 10.6, "scale": 0.4, "low": 9.5, "high": 12.0}
    assert mapping["log_host_aperture_scale"] == {"dist": "Normal", "loc": 0.0, "scale": 0.5}
    assert mapping["log_sfh_age_gyr"] == {"dist": "Normal", "loc": np.log(7.0), "scale": 0.3}
    assert mapping["log_sfh_tau_over_age"] == {"dist": "Normal", "loc": np.log(0.25), "scale": 0.3}
    assert mapping["gal_lgmet"] == {"dist": "Normal", "loc": 0.0, "scale": 0.2}
    assert mapping["log_gal_lgmet_scatter"] == {"dist": "Normal", "loc": np.log(0.1), "scale": 0.3}
    assert mapping["host_template_age_prior"] == {"type": "prefer_old", "pivot_gyr": 1.0, "strength": 2.0}
    assert mapping["host_sfh_model"] == "delayed"
    assert np.isclose(mapping["log_cont_norm"]["loc"], np.log(2.0 * np.median(np.abs(flux))))


def test_prior_config_coerces_nested_semantic_sections():
    prior = PriorConfig(
        continuum={"powerlaw": {"slope": dist.Normal(loc=-1.2, scale=0.2)}},
        feii={"uv_norm": dist.Normal(loc=-5.0, scale=0.1)},
        lines={"dmu_scale_mult": 0.1, "sig_scale_mult": 0.2, "amp_scale_mult": 0.3},
        host={"sfh_model": "delayed", "stellar_mass": dist.Normal(loc=10.5, scale=0.2)},
    )
    mapping = prior.to_mapping()

    assert mapping["PL_slope"] == {"dist": "Normal", "loc": -1.2, "scale": 0.2}
    assert mapping["log_Fe_uv_norm"] == {"dist": "Normal", "loc": -5.0, "scale": 0.1}
    assert mapping["line_dmu_scale_mult"] == 0.1
    assert mapping["line_sig_scale_mult"] == 0.2
    assert mapping["line_amp_scale_mult"] == 0.3
    assert mapping["host_sfh_model"] == "delayed"
    assert mapping["log_stellar_mass"] == {"dist": "Normal", "loc": 10.5, "scale": 0.2}


def test_fit_config_coerces_bal_mapping():
    cfg = FitConfig(
        observation=Observation(redshift=0.1),
        spectroscopy=SpectroscopyData(wave_obs=[4000.0, 5000.0], fluxes=[1.0, 1.1]),
        bal={
            "enabled": True,
            "tau_scale": 0.4,
            "covering_loc": 0.2,
            "covering_scale": 0.1,
            "covering_high": 0.8,
        }
    )

    assert isinstance(cfg.bal, BALConfig)
    assert cfg.bal.enabled is True
    assert np.isclose(cfg.bal.tau_scale, 0.4)
    assert np.isclose(cfg.bal.covering_loc, 0.2)
    assert np.isclose(cfg.bal.covering_scale, 0.1)
    assert np.isclose(cfg.bal.covering_high, 0.8)


def test_fit_config_coerces_prior_config_mapping():
    cfg = FitConfig(
        observation=Observation(redshift=0.1),
        spectroscopy=SpectroscopyData(wave_obs=[4000.0, 5000.0], fluxes=[1.0, 1.1]),
        host=HostConfig(sfh_model="flexible"),
        prior_config={"continuum": {"power_law_pivot": 2500.0}},
    )

    assert isinstance(cfg.prior_config, PriorConfig)
    assert cfg.host.sfh_model == "flexible"
    assert cfg.prior_config.to_mapping()["PL_pivot"] == 2500.0


def test_build_default_prior_config_has_expected_keys():
    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    cfg = build_default_prior_config(flux)
    mapping = cfg.to_mapping()

    required = [
        'log_cont_norm',
        'PL_norm',
        'PL_slope',
        'PL_pivot',
        'poly_pivot',
        'log_reddening_a2500',
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
        assert k in mapping
    assert isinstance(cfg, PriorConfig)
    assert mapping["poly_pivot"] is None
    assert mapping["log_stellar_mass"] == {
        "dist": "TruncatedNormal",
        "loc": 9.0,
        "scale": 0.75,
        "low": 7.0,
        "high": 12.0,
    }
    assert mapping["log_Balmer_vel"] == {
        "dist": "TruncatedNormal",
        "loc": np.log(3000.0),
        "scale": 0.3,
        "low": np.log(1000.0),
        "high": np.log(15000.0),
    }
    assert mapping["log_host_aperture_scale"] == {"dist": "Normal", "loc": 0.0, "scale": 0.5}
    assert mapping["host_template_age_prior"] == {
        "type": "prefer_old",
        "pivot_gyr": 1.0,
        "strength": 1.0,
        "min_logit": -3.0,
        "max_logit": 2.0,
    }
    assert mapping["log_sfh_tau_over_age"] == {"dist": "Normal", "loc": 0.0, "scale": 0.5}
    assert mapping["log_gal_sigma_kms"]["dist"] == "Normal"
    assert mapping["log_reddening_a2500"] == {"dist": "Normal", "loc": np.log(0.1), "scale": 0.6}


def test_build_default_prior_config_scales_with_flux_median():
    flux = np.array([10.0, 20.0, 30.0], dtype=float)
    cfg = build_default_prior_config(flux)

    expected = np.log(np.median(np.abs(flux)))
    mapping = cfg.to_mapping()

    got = float(mapping['log_cont_norm']['loc'])
    assert np.isfinite(got)
    assert np.isclose(got, expected)
    assert mapping['log_cont_norm']['dist'] == 'LogNormal'
    assert mapping['PL_slope']['dist'] == 'Normal'
    assert mapping['tau_host']['dist'] == 'HalfNormal'


def test_build_default_prior_config_accepts_manual_pl_pivot():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    cfg = build_default_prior_config(flux, pl_pivot=3000.0)
    mapping = cfg.to_mapping()
    assert mapping["PL_pivot"] == 3000.0


def test_build_default_prior_config_uses_explicit_dist_fields():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    cfg = build_default_prior_config(flux)
    mapping = cfg.to_mapping()

    assert mapping["log_frac_host"]["dist"] == "StudentT"
    assert mapping["host_redshift_prior"]["enabled"] is False
    assert mapping["host_redshift_prior"]["z_mid"] == 1.0
    assert mapping["host_redshift_prior"]["width"] == 0.2
    assert mapping["host_redshift_prior"]["lowz_loc_offset"] == 0.0
    assert mapping["host_redshift_prior"]["highz_loc_offset"] == -8.0
    assert mapping["host_redshift_prior"]["lowz_scale_mult"] == 1.0
    assert mapping["host_redshift_prior"]["highz_scale_mult"] == 0.05
    assert mapping["host_redshift_prior"]["lowz_df"] == 3.0
    assert mapping["host_redshift_prior"]["highz_df"] == 20.0
    assert mapping["log_Fe_uv_norm"]["dist"] == "LogNormal"
    assert np.isclose(mapping["log_Fe_uv_norm"]["loc"], np.log(0.03 * 2.0))
    assert mapping["log_Fe_uv_norm"]["scale"] == 1.0
    assert mapping["log_Fe_op_over_uv"] == {"dist": "Normal", "loc": 0.0, "scale": 1.0}
    assert mapping["log_Fe_uv_FWHM"]["dist"] == "LogNormal"
    assert np.isclose(mapping["log_Fe_uv_FWHM"]["loc"], np.log(3000.0))
    assert mapping["log_Fe_uv_FWHM"]["scale"] == 0.5
    assert mapping["log_Fe_op_FWHM"]["dist"] == "LogNormal"
    assert np.isclose(mapping["log_Fe_op_FWHM"]["loc"], np.log(3000.0))
    assert mapping["log_Fe_op_FWHM"]["scale"] == 0.5
    assert mapping["Fe_uv_shift"]["dist"] == "Normal"
    assert mapping["frac_jitter"]["dist"] == "HalfNormal"
    assert mapping["add_jitter"]["dist"] == "HalfNormal"


def test_build_default_bal_components_exposes_common_bal_lines():
    comps = build_default_bal_components(np.array([1.0, 2.0, 3.0], dtype=float))

    names = [comp.name for comp in comps]
    assert names == ["bal_nv", "bal_siiv", "bal_civ"]
    assert comps[2].metadata["component_type"] == "bal_absorption"
    assert comps[2].metadata["line_lambda"] == 1549.06
    assert comps[2].metadata["shared_parameter_sites"]["v_out"] == "custom_bal_v_out"
    assert comps[2].metadata["shared_parameter_sites"]["tau_peak"] == "custom_bal_tau_peak"
    assert comps[2].metadata["shared_parameter_sites"]["covering"] == "custom_bal_covering"
    assert comps[2].metadata["shared_parameter_sites"]["fwhm_kms"] == "custom_bal_fwhm_kms"
    assert np.isclose(comps[0].parameter_priors["tau_peak"]["scale"], 0.25)
    assert np.isclose(comps[1].parameter_priors["tau_peak"]["scale"], 0.25)
    tau_cfg = comps[2].parameter_priors["tau_peak"]
    assert tau_cfg["dist"] == "HalfNormal"
    assert np.isclose(tau_cfg["scale"], 0.25)
    covering_cfg = comps[2].parameter_priors["covering"]
    assert covering_cfg["dist"] == "TruncatedNormal"
    assert covering_cfg["loc"] == 0.15
    assert covering_cfg["scale"] == 0.12
    assert covering_cfg["low"] == 0.0
    assert covering_cfg["high"] == 0.70
    v_out_cfg = comps[2].parameter_priors["v_out"]
    assert v_out_cfg["dist"] == "TruncatedNormal"
    assert v_out_cfg["loc"] == 6000.0
    assert v_out_cfg["scale"] == 2500.0
    assert v_out_cfg["low"] == 3000.0
    assert v_out_cfg["high"] == 12000.0
    fwhm_cfg = comps[2].parameter_priors["fwhm_kms"]
    assert fwhm_cfg["dist"] == "TruncatedNormal"
    assert fwhm_cfg["loc"] == 8000.0
    assert fwhm_cfg["scale"] == 2500.0
    assert fwhm_cfg["low"] == 2000.0
    assert fwhm_cfg["high"] == 15000.0
    shape_cfg = comps[2].parameter_priors["shape_power"]
    assert shape_cfg["dist"] == "TruncatedNormal"
    assert shape_cfg["loc"] == 2.0
    assert shape_cfg["low"] == 2.0
    assert shape_cfg["high"] == 12.0


def test_build_default_bal_components_accepts_prior_overrides():
    comps = build_default_bal_components(
        np.array([1.0, 2.0, 3.0], dtype=float),
        tau_scale=0.6,
        covering_loc=0.4,
        covering_scale=0.1,
        covering_high=0.85,
        fwhm_kms_loc=6000.0,
        fwhm_kms_scale=1500.0,
        fwhm_kms_low=2500.0,
        fwhm_kms_high=12000.0,
    )

    for comp in comps:
        assert np.isclose(comp.parameter_priors["tau_peak"]["scale"], 0.6)
        covering_cfg = comp.parameter_priors["covering"]
        assert np.isclose(covering_cfg["loc"], 0.4)
        assert np.isclose(covering_cfg["scale"], 0.1)
        assert np.isclose(covering_cfg["high"], 0.85)
        fwhm_cfg = comp.parameter_priors["fwhm_kms"]
        assert np.isclose(fwhm_cfg["loc"], 6000.0)
        assert np.isclose(fwhm_cfg["scale"], 1500.0)
        assert np.isclose(fwhm_cfg["low"], 2500.0)
        assert np.isclose(fwhm_cfg["high"], 12000.0)


def test_default_line_table_contains_expanded_uv_complexes():
    cfg = build_default_prior_config(np.array([1.0, 2.0, 3.0], dtype=float))
    rows = cfg.to_mapping()["line"]["table"]
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
        "SiIV_OIV1_br",
        "SiIV_OIV2_br",
        "CII1335",
        "OI1304",
        "Lya_br",
        "NV1240_br",
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
    rows = cfg.to_mapping()["line"]["table"]
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


def test_default_oiii_doublets_are_tied_with_physical_ratio():
    cfg = build_default_prior_config(np.array([1.0, 2.0, 3.0], dtype=float))
    by_name = {row["linename"]: row for row in cfg.to_mapping()["line"]["table"]}

    assert by_name["OIII4959c"]["findex"] == by_name["OIII5007c"]["findex"]
    assert by_name["OIII4959w"]["findex"] == by_name["OIII5007w"]["findex"]
    assert by_name["OIII4959c"]["findex"] != by_name["OIII4959w"]["findex"]
    assert np.isclose(
        by_name["OIII5007c"]["fvalue"] * by_name["OIII5007c"]["lambda"]
        / (by_name["OIII4959c"]["fvalue"] * by_name["OIII4959c"]["lambda"]),
        2.98,
    )
    assert np.isclose(
        by_name["OIII5007w"]["fvalue"] * by_name["OIII5007w"]["lambda"]
        / (by_name["OIII4959w"]["fvalue"] * by_name["OIII4959w"]["lambda"]),
        2.98,
    )


def test_combined_optional_config_preserves_oi_doublet_ratio():
    cfg = build_default_prior_config(
        np.array([1.0, 2.0, 3.0], dtype=float),
        include_elg_narrow_lines=True,
    )
    by_name = {row["linename"]: row for row in cfg.to_mapping()["line"]["table"]}

    assert np.isclose(
        by_name["OI6300"]["fvalue"] * by_name["OI6300"]["lambda"]
        / (by_name["OI6363"]["fvalue"] * by_name["OI6363"]["lambda"]),
        3.05,
    )
