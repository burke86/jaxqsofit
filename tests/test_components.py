import numpy as np
import jax
from numpyro.handlers import seed, substitute, trace

from jaxqsofit.components import SpectralComponentConfig, evaluate_joint_spectral_components
from jaxqsofit.defaults import _build_default_prior_config as build_default_prior_config


def test_evaluate_joint_spectral_components_uses_external_continuum():
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(3))).get_trace(
        wave_obs=wave_obs,
        redshift=0.1,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            use_multiplicative_tilt=False,
        ),
    )

    assert np.allclose(np.asarray(tr["jqf_total_model"]["value"]), continuum)
    assert np.allclose(np.asarray(tr["jqf_line_model"]["value"]), 0.0)


def test_evaluate_joint_spectral_components_adds_line_sites():
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(4))).get_trace(
        wave_obs=wave_obs,
        redshift=0.1,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_centers_rest=(4861.33,),
            line_names=("Hbeta",),
            broad_line_names=("Hbeta",),
        ),
    )

    assert "jqf_line_amp_Hbeta" in tr
    assert "jqf_line_fwhm_Hbeta" in tr
    assert "jqf_line_velocity_Hbeta" in tr
    assert np.asarray(tr["jqf_total_model"]["value"]).shape == wave_obs.shape


def test_evaluate_joint_spectral_components_uses_default_tied_lines():
    wave_obs = np.linspace(4700.0, 5100.0, 96)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(5))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_flux_scale_mjy=2.0,
        ),
    )

    assert "jqf_line_dmu_independent_group_std" in tr
    assert "jqf_line_log_fwhm_delta_group_std" in tr
    assert "jqf_line_amp_group" in tr
    assert tr["jqf_line_dmu_group"]["type"] == "deterministic"
    assert tr["jqf_line_sig_group"]["type"] == "deterministic"
    assert tr["jqf_line_amp_group"]["type"] == "deterministic"
    assert any(
        name.startswith("jqf_line_amp_") and name != "jqf_line_amp_group"
        for name in tr
    )
    assert "jqf_line_amp_per_component" in tr
    assert "jqf_line_model_broad" in tr
    assert "jqf_line_model_narrow" in tr
    assert np.asarray(tr["jqf_total_model"]["value"]).shape == wave_obs.shape


def test_evaluate_joint_spectral_components_filters_tied_lines_to_coverage():
    wave_obs = np.linspace(4850.0, 5010.0, 64)
    continuum = np.full_like(wave_obs, 1.0)
    line_table = [
        {
            "lambda": 4862.68,
            "linename": "Hb",
            "compname": "Hb",
            "minwav": 4800.0,
            "maxwav": 4920.0,
            "inisca": 0.1,
            "minsca": 1.0e-4,
            "maxsca": 1.0,
            "inisig": 1.0e-3,
            "minsig": 1.0e-4,
            "maxsig": 1.0e-2,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        },
        {
            "lambda": 6564.61,
            "linename": "Ha",
            "compname": "Ha",
            "minwav": 6400.0,
            "maxwav": 6800.0,
            "inisca": 0.1,
            "minsca": 1.0e-4,
            "maxsca": 1.0,
            "inisig": 1.0e-3,
            "minsig": 1.0e-4,
            "maxsig": 1.0e-2,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        },
    ]

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(8))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_table=line_table,
            line_coverage_rest=(4800.0, 5050.0),
        ),
    )

    assert np.asarray(tr["jqf_line_amp_per_component"]["value"]).shape == (1,)


def test_evaluate_joint_spectral_components_accepts_prior_config_object_as_line_prior_config():
    wave_obs = np.linspace(4700.0, 5100.0, 96)
    continuum = np.full_like(wave_obs, 2.0)
    prior_config = build_default_prior_config(
        continuum,
        include_elg_narrow_lines=False,
        include_high_ionization_lines=False,
    )

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(6))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_prior_config=prior_config,
        ),
    )

    assert "jqf_line_dmu_independent_group_std" in tr
    assert tr["jqf_line_dmu_group"]["type"] == "deterministic"
    assert "jqf_line_amp_per_component" in tr
    assert np.asarray(tr["jqf_total_model"]["value"]).shape == wave_obs.shape


def test_evaluate_joint_spectral_components_reports_fixed_narrow_line_controls():
    wave_obs = np.linspace(4990.0, 5010.0, 96)
    continuum = np.full_like(wave_obs, 1.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(6))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_tied_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            line_centers_rest=(5000.0,),
            line_names=("OIII5007c",),
            fixed_narrow_fwhm_kms=321.0,
            fixed_narrow_amp_scale=2.5,
        ),
    )

    assert tr["jqf_line_narrow_fwhm_kms"]["value"] == 321.0
    assert tr["jqf_line_narrow_amp_scale"]["value"] == 2.5
    assert np.nanmax(np.asarray(tr["jqf_line_model_narrow"]["value"])) > 0.0


def test_evaluate_joint_spectral_components_converts_feii_template_to_fnu_shape():
    wave_obs = np.array([2000.0, 3000.0, 4000.0])
    continuum = np.zeros_like(wave_obs)
    template_wave = np.array([1000.0, 5000.0])
    template_flux = np.ones_like(template_wave)
    fn = substitute(
        seed(evaluate_joint_spectral_components, jax.random.PRNGKey(7)),
        data={
            "jqf_feii_norm": 1.0,
            "jqf_feii_fwhm": 1.0,
            "jqf_feii_shift": 0.0,
        },
    )

    tr = trace(fn).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            use_feii=True,
            use_balmer_continuum=False,
            feii_fnu_pivot_rest=3000.0,
        ),
        feii_template_wave_rest=template_wave,
        feii_template_flux=template_flux,
    )
    feii = np.asarray(tr["jqf_feii_model"]["value"])

    assert feii[2] / feii[0] > 3.9
    assert np.isclose(feii[1] / feii[0], (3000.0 / 2000.0) ** 2, rtol=0.05)
