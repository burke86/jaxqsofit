import os
import h5py
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import numpyro
import numpyro.distributions as dist

import jaxqsofit
import jaxqsofit.core as coremod
import jaxqsofit.model as modelmod
import jaxqsofit.plotting as plottingmod
from jaxqsofit import JAXQSOFit
from jaxqsofit.config import PriorConfig
from jaxqsofit.defaults import _build_default_prior_config as build_default_prior_config
from jaxqsofit.results import FitResult, PredictionResult


def _make_simple_spectrum(n=64):
    lam = np.linspace(3800.0, 9200.0, n)
    flux = 50.0 + 0.002 * (lam - 6000.0)
    err = np.full_like(flux, 0.5)
    return lam, flux, err


def _make_wide_spectrum(n=256):
    lam = np.linspace(3000.0, 10000.0, n)
    flux = 40.0 + 0.0015 * (lam - 6000.0)
    err = np.full_like(flux, 0.4)
    return lam, flux, err


def _fake_spectral_result_inputs():
    wave = np.array([4800.0, 4862.68, 5008.24])
    raw = {
        "line_model": np.full((3, 3), 2.0),
        "continuum_model": np.full((3, 3), 10.0),
        "line_amp_per_component": np.array(
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]]
        ),
        "line_mu_per_component": np.log(
            np.array(
                [
                    [4862.68, 4863.0, 5008.24],
                    [4862.8, 4863.1, 5008.3],
                    [4862.9, 4863.2, 5008.4],
                ]
            )
        ),
        "line_sig_per_component": np.full((3, 3), 0.01),
    }
    fitter = SimpleNamespace(
        wave=wave,
        flux=np.array([11.0, 12.0, 13.0]),
        err=np.full(3, 0.5),
        tied_line_meta={
            "names": ["Hb_br_1", "Hb_br_2", "OIII_5007_1"],
            "line_lambda": np.array([4862.68, 4862.68, 5008.24]),
            "broad_mask": np.array([True, True, False]),
        },
    )
    fitter._ensure_posterior_state = lambda: SimpleNamespace(predictive=raw)
    data = {
        "wave": wave,
        "draws": {
            "continuum": np.full((2, 3), 10.0),
            "PL": np.full((2, 3), 6.0),
            "Fe_uv": np.full((2, 3), 1.0),
            "Fe_op": np.full((2, 3), 2.0),
            "Balmer_cont": np.full((2, 3), 0.5),
            "host": np.full((2, 3), 0.5),
        },
    }
    return fitter, data


def test_prediction_spectrum_has_named_unit_explicit_line_results():
    fitter, data = _fake_spectral_result_inputs()

    spectrum = PredictionResult(data, fitter).spectrum

    assert tuple(spectrum.lines) == ("Hb_br_1", "Hb_br_2", "OIII_5007")
    assert spectrum.line_groups["Hb_br"].component_names == (
        "Hb_br_1",
        "Hb_br_2",
    )
    hb1 = spectrum.lines["Hb_br_1"]
    assert hb1.parent_line == "Hb_br"
    assert hb1.component_index == 1
    assert hb1.kind == "broad"
    assert hb1.amplitude_flambda_1e17.shape == (2,)
    assert np.allclose(hb1.center_rest_angstrom, [4862.68, 4862.8])
    assert np.allclose(
        spectrum.line_groups["Hb_br"].total_flux_erg_s_cm2,
        spectrum.lines["Hb_br_1"].flux_erg_s_cm2
        + spectrum.lines["Hb_br_2"].flux_erg_s_cm2,
    )
    assert spectrum.model_flambda_1e17.shape == (2, 3)
    assert np.allclose(spectrum.model_flambda_1e17, 12.0)
    assert np.allclose(spectrum.feii_flambda_1e17, 3.0)
    assert np.array_equal(spectrum.mask, np.ones(3, dtype=bool))


def test_fit_result_spectrum_requests_the_native_grid_and_is_cached():
    fitter, data = _fake_spectral_result_inputs()
    calls = []

    def reconstruct(**kwargs):
        calls.append(kwargs)
        return data

    fitter.reconstruct_posterior_spectrum = reconstruct
    result = FitResult(
        fitter=fitter,
        samples={},
        median={},
        method="test",
    )

    first = result.spectrum
    second = result.spectrum

    assert first is second
    assert len(calls) == 1
    assert np.array_equal(calls[0]["wave_out"], fitter.wave)


def test_calculate_sn_skips_uncovered_standard_windows_without_warnings():
    wave = np.linspace(2990.0, 3060.0, 96)
    flux = 20.0 + 0.03 * np.sin(wave / 3.0)
    fitter = object.__new__(JAXQSOFit)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sn_ratio = fitter._calculate_sn(wave, flux)

    assert np.isfinite(sn_ratio)
    assert sn_ratio > 0.0


def test_calculate_sn_constant_fallback_has_no_divide_by_zero_warning():
    wave = np.linspace(7000.0, 7100.0, 64)
    flux = np.full_like(wave, 20.0)
    fitter = object.__new__(JAXQSOFit)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sn_ratio = fitter._calculate_sn(wave, flux)

    assert sn_ratio == -1.0


def _build_bundle_source(tmp_path, filename, decompose_host):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        ra=150.0,
        dec=2.0,
        filename=filename,
        output_path=str(tmp_path),
    )
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q._fit_prior_config = build_default_prior_config(flux).to_mapping()
    q._fit_prior_config["host_sfh_model"] = "flexible"
    q._fit_fsps_age_grid = (0.1, 1.0)
    q._fit_fsps_logzsol_grid = (-0.5, 0.0)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q._fit_fit_lines = False
    q._fit_decompose_host = bool(decompose_host)
    q._fit_fit_pl = True
    q._fit_fit_fe = False
    q._fit_fit_bc = False
    q._fit_fit_poly = False
    q._fit_fit_poly_order = 2
    q._fit_fit_reddening = False
    q._fit_use_psf_phot = False
    q._fit_custom_components = ()
    q._fit_custom_line_components = ()
    q.numpyro_samples = {
        "cont_norm": np.array([1.0, 1.1, 0.9]),
        "log_frac_host": np.array([0.0, 0.1, -0.1]),
        "PL_norm": np.array([1.0, 1.0, 1.0]),
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
    }
    if decompose_host:
        q.numpyro_samples.update({
            "tau_host": np.array([1.0, 1.0, 1.0]),
            "fsps_weights_raw": np.zeros((3, 4)),
            "gal_v_kms": np.zeros(3),
            "gal_sigma_kms": np.full(3, 100.0),
        })
    q.save_fig = False
    return q, lam, flux, err


def test_init_err_optional_defaults_to_small_value():
    lam, flux, _ = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, z=0.1)

    assert q.err_in.shape == flux.shape
    assert np.allclose(q.err_in, 1e-6)


def test_init_psf_defaults_band_labels():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        psf_mags=np.array([20.0, 19.8, 19.6]),
        psf_mag_errs=np.array([0.1, 0.1, 0.1]),
    )

    assert q.psf_bands == ["u", "g", "r"]


def test_predictive_return_sites_include_requested_continuum_luminosities():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q._fit_prior_config = build_default_prior_config(flux)
    q.L_conti_wave = np.array([1350.0, 3000.0, 5100.0], dtype=float)

    sites = q._predictive_return_sites()

    assert "log_lambda_Llambda_1350_agn" in sites
    assert "log_lambda_Llambda_2500_agn" in sites
    assert "log_lambda_Llambda_3000_agn" in sites
    assert "log_lambda_Llambda_5100_agn" in sites
    assert "spectral_likelihood_weight" in sites
    assert "frac_jitter" in sites
    assert "frac_fe_jitter" in sites
    assert "add_jitter" in sites
    assert "line_component_profiles" in sites
    assert "psf_model" not in sites


def test_predictive_return_sites_prune_inactive_line_and_psf_outputs():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q._fit_fit_lines = False
    q._fit_fit_fe = False
    q._fit_use_psf_phot = False

    sites = q._predictive_return_sites()

    assert "line_model" in sites
    assert "line_component_profiles" not in sites
    assert "line_amp_per_component" not in sites
    assert "psf_model" not in sites
    assert "line_component_profiles_psf" not in sites
    assert "frac_jitter" in sites
    assert "frac_fe_jitter" not in sites
    assert "add_jitter" in sites


def test_numpyro_geometry_reparam_config_only_decenters_flexible_host_for_nuts():
    prior_config = {
        "PL_slope": {"dist": "TruncatedNormal", "loc": -1.5, "scale": 0.3, "low": -3.5, "high": 0.5},
        "log_Fe_uv_norm": {"dist": "Normal", "loc": -5.0, "scale": 0.2},
        "log_Fe_uv_FWHM": {"dist": "LogNormal", "loc": np.log(3000.0), "scale": 0.3},
        "gal_v_kms": {"dist": "Normal", "loc": 0.0, "scale": 150.0},
    }

    delayed_config = coremod._numpyro_geometry_reparam_config(
        {**prior_config, "host_sfh_model": "delayed"},
        fit_pl=True,
        fit_fe=True,
        fit_bc=False,
        decompose_host=True,
    )
    flexible_config = coremod._numpyro_geometry_reparam_config(
        {**prior_config, "host_sfh_model": "flexible"},
        fit_pl=True,
        fit_fe=True,
        fit_bc=False,
        decompose_host=True,
    )

    assert delayed_config == {}
    assert set(flexible_config) == {"fsps_weights_raw"}
    assert isinstance(
        flexible_config["fsps_weights_raw"], coremod.LocScaleReparam
    )


def test_line_table_kind_filter_removes_broad_rows_only():
    flux = np.array([1.0, 2.0, 3.0], dtype=float)
    prior = build_default_prior_config(flux).to_mapping()

    filtered = coremod._filter_prior_line_table_by_kind(
        prior,
        use_broad_lines=False,
        use_narrow_lines=True,
    )
    names = [row["linename"] for row in filtered["line"]["table"]]

    assert names
    assert not any("_br" in name.lower() for name in names)
    assert any(name.endswith("_na") or "oiii" in name.lower() for name in names)


def test_custom_line_component_kind_filter_removes_broad_components():
    def _dummy_eval(wave, params, metadata):
        return np.zeros_like(wave)

    broad = coremod.CustomLineComponentSpec(
        name="broad_wing",
        parameter_priors={"amp": {"dist": "Normal", "loc": 1.0, "scale": 0.1}},
        evaluate=_dummy_eval,
        line_kind="broad",
    )
    narrow = coremod.CustomLineComponentSpec(
        name="narrow_core",
        parameter_priors={"amp": {"dist": "Normal", "loc": 1.0, "scale": 0.1}},
        evaluate=_dummy_eval,
        line_kind="narrow",
    )

    filtered = coremod._filter_custom_line_components_by_kind(
        (broad, narrow),
        use_broad_lines=False,
        use_narrow_lines=True,
    )

    assert tuple(comp.name for comp in filtered) == ("narrow_core",)


def test_spectral_likelihood_weight_from_resolving_power():
    wave = np.arange(5000.0, 5010.0, 1.0)

    weight = float(modelmod.spectral_likelihood_weight_from_resolving_power(wave, 2000.0))

    assert weight == pytest.approx(0.36, rel=1.0e-2)
    assert float(modelmod.spectral_likelihood_weight_from_resolving_power(wave, None)) == 1.0


def test_from_arrays_stores_resolving_power():
    lam, flux, err = _make_simple_spectrum()

    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1, resolving_power=2000.0)

    assert q.resolving_power == 2000.0
    assert q.config.spectroscopy.resolving_power == 2000.0


def test_prepare_psf_photometry_masks_invalid_and_builds_transmissions():
    lam, flux, err = _make_wide_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    mags, mag_errs, bands, filt_curves, use_psf = q._prepare_psf_photometry(
        wave_obs=lam,
        psf_mags=np.array([20.0, 19.9, np.nan, 19.5]),
        psf_mag_errs=np.array([0.10, 0.12, 0.20, -1.0]),
        psf_bands=["u", "g", "r", "i"],
        use_psf_phot=True,
    )

    assert use_psf is True
    assert bands == ["u", "g"]
    assert mags.shape == (2,)
    assert mag_errs.shape == (2,)
    assert filt_curves["trans"].shape == (2, lam.size)
    assert np.all(filt_curves["trans"] >= 0.0)


def test_prepare_psf_photometry_dereddens_psf_mags_bandpass_consistently():
    lam, flux, err = _make_wide_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q._fit_deredden = True
    q.ebv_mw = 0.12

    mags, mag_errs, bands, filt_curves, use_psf = q._prepare_psf_photometry(
        wave_obs=lam,
        psf_mags=np.array([20.0, 19.8, 19.6]),
        psf_mag_errs=np.array([0.10, 0.10, 0.10]),
        psf_bands=["u", "g", "r"],
        use_psf_phot=True,
    )

    assert use_psf is True
    assert bands == ["u", "g", "r"]
    assert mags.shape == (3,)
    assert mag_errs.shape == (3,)
    assert np.allclose(q.psf_mags_raw, np.array([20.0, 19.8, 19.6]))
    assert np.allclose(q.psf_mag_errs_raw, np.array([0.10, 0.10, 0.10]))
    assert np.allclose(q.psf_mags_dered, mags)
    assert np.allclose(q.psf_mag_errs_dered, mag_errs)
    assert np.all(mags < q.psf_mags_raw)
    assert (q.psf_mags_raw[0] - mags[0]) > (q.psf_mags_raw[-1] - mags[-1])
    assert filt_curves["trans"].shape == (3, lam.size)


def test_prepare_psf_photometry_zero_ebv_keeps_mags_unchanged():
    lam, flux, err = _make_wide_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q._fit_deredden = True
    q.ebv_mw = 0.0

    mags, mag_errs, bands, _filt_curves, use_psf = q._prepare_psf_photometry(
        wave_obs=lam,
        psf_mags=np.array([20.0, 19.8]),
        psf_mag_errs=np.array([0.10, 0.12]),
        psf_bands=["g", "r"],
        use_psf_phot=True,
    )

    assert use_psf is True
    assert bands == ["g", "r"]
    assert np.allclose(mags, np.array([20.0, 19.8]))
    assert np.allclose(q.psf_mags_raw, mags)
    assert np.allclose(q.psf_mags_dered, mags)
    assert np.allclose(q.psf_mag_errs_dered, mag_errs)


def test_de_redden_invalid_placeholder_coordinates_raise_clear_error():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    with pytest.raises(ValueError, match="fit\\(deredden=False\\)|valid sky coordinates"):
        q._validate_deredden_coordinates(ra=-999, dec=-999)


def test_fit_rejects_legacy_keywords_with_config_hint():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    with pytest.raises(TypeError, match="configuration-first.*FitConfig.*fit_lines"):
        q.fit(fit_lines=False)


def test_build_fsps_grid_for_fit_skips_template_load_when_host_disabled(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    def _boom(**kwargs):
        raise AssertionError("FSPS templates should not be loaded when decompose_host=False")

    monkeypatch.setattr(coremod, "build_fsps_template_grid", _boom)

    grid = q._build_fsps_grid_for_fit(
        wave=lam,
        age_grid_gyr=(0.1, 1.0),
        logzsol_grid=(-0.5, 0.0),
        dsps_ssp_fn="missing.h5",
        decompose_host=False,
    )

    assert grid.templates.shape == (lam.size, 4)
    assert np.allclose(grid.templates, 0.0)
    assert [(meta["logzsol"], meta["tage_gyr"]) for meta in grid.template_meta] == [
        (-0.5, 0.1),
        (-0.5, 1.0),
        (0.0, 0.1),
        (0.0, 1.0),
    ]


def test_fit_dispatch_nuts(monkeypatch):
    lam, flux, err = _make_wide_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {'nuts': 0, 'kwargs': None}

    def _stub_nuts(**kwargs):
        called['nuts'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_numpyro_fit', _stub_nuts)

    q.config.inference.method = 'nuts'
    q.config.inference.dense_mass = False
    q.config.inference.max_tree_depth = 6
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.config.psf_photometry = coremod.PSFPhotometryData(
        magnitudes=np.array([19.8, 19.6]),
        magnitude_errors=np.array([0.05, 0.06]),
        filter_names=["g", "r"],
    )
    result = q.fit()

    assert isinstance(result, FitResult)
    assert result.method == "nuts"
    assert called['nuts'] == 1
    assert called['kwargs']['use_psf_phot'] is True
    assert called['kwargs']['dense_mass'] is False
    assert called['kwargs']['max_tree_depth'] == 6
    assert called['kwargs']['psf_mags'].shape == (2,)
    assert called['kwargs']['psf_filter_curves']['trans'].shape == (2, q.lam.size)


def test_line_complex_dense_mass_blocks_group_local_latents():
    tied_line_meta = {
        "amp_complex_groups": [
            {"complex_index": 0, "fgroup_ids": [0, 1, 2]},
            {"complex_index": 1, "fgroup_ids": [3]},
        ],
        "broad_width_order_complex_indices": [0],
        "broad_width_order_site_labels": ["Hb"],
        "broad_width_order_groups": [np.array([0])],
        "n_wgroups": 2,
        "wgroup": np.array([0, 1]),
        "broad_mask": np.array([1.0, 0.0]),
        "unordered_width_group_ids": np.array([0, 1]),
        "independent_vgroup_ids": np.array([0, 1]),
        "nlr_vgroup_families": {
            "low": np.array([2]),
            "high": np.array([3]),
            "coronal": np.array([4]),
        },
        "direct_width_groups": [
            {"group_id": 2, "site_label": "OIII_wing"},
        ],
        "nlr_wgroup_families": {
            "low": np.array([3]),
            "high": np.array([4]),
            "coronal": np.array([5]),
        },
        "broad_centroid_hierarchy_groups": [
            {"complex_index": 0, "component_groups": [0, 1]}
        ],
    }

    blocks = coremod._line_complex_dense_mass_blocks(
        tied_line_meta, standardized_amplitudes=True
    )

    assert blocks == [
        ("line_amp_complex_1_std",),
        (
            "line_amp_complex_0_std",
            "line_broad_center_0_std",
            "line_broad_relative_offsets_0_std",
            "line_ordered_width_logits_Hb_std",
        ),
    ]


def test_ordered_line_complexes_use_separate_dense_blocks():
    tied_line_meta = {
        "amp_complex_groups": [
            {"complex_index": 0, "site_label": "Hb"},
            {"complex_index": 1, "site_label": "MgII"},
        ],
        "broad_width_order_complex_indices": [0, 1],
        "broad_width_order_site_labels": ["Hb", "MgII"],
        "broad_centroid_hierarchy_groups": [
            {"complex_index": 0},
            {"complex_index": 1},
        ],
    }

    blocks = coremod._line_complex_dense_mass_blocks(
        tied_line_meta, standardized_amplitudes=True
    )

    assert blocks == [
        (
            "line_amp_Hb_std",
            "line_broad_center_0_std",
            "line_broad_relative_offsets_0_std",
            "line_ordered_width_logits_Hb_std",
        ),
        (
            "line_amp_MgII_std",
            "line_broad_center_1_std",
            "line_broad_relative_offsets_1_std",
            "line_ordered_width_logits_MgII_std",
        ),
    ]


def test_nuts_transition_diagnostics_distinguish_final_level_from_full_tree():
    diagnostics = coremod.summarize_nuts_transition_fields(
        {
            "num_steps": np.array([[31, 128, 254, 255]], dtype=float),
            "accept_prob": np.array([[0.8, 0.9, 0.95, 1.0]], dtype=float),
            "diverging": np.array([[False, False, True, False]]),
            "energy": np.array([[1.0, 2.0, 1.5, 2.5]], dtype=float),
        },
        max_tree_depth=8,
    )

    assert diagnostics["final_tree_level_fraction"] == pytest.approx(0.75)
    assert diagnostics["full_trajectory_fraction"] == pytest.approx(0.25)
    assert diagnostics["max_num_steps_fraction"] == pytest.approx(0.25)
    assert diagnostics["n_max_num_steps"] == 1
    assert diagnostics["max_num_steps"] == 255
    assert diagnostics["n_divergent"] == 1
    assert diagnostics["bfmi"].shape == (1,)


def test_optax_state_reset_removes_stale_nuts_diagnostics():
    lam, flux, err = _make_simple_spectrum()
    fitter = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    nuts_state_names = (
        "numpyro_mcmc",
        "nuts_mass_matrix_structure",
        "nuts_extra_fields",
        "nuts_diagnostics",
        "nuts_metric_diagnostics",
    )
    for name in nuts_state_names:
        setattr(fitter, name, object())

    fitter._clear_nuts_run_state()

    assert not any(name in fitter.__dict__ for name in nuts_state_names)


def test_nuts_metric_diagnostics_flag_nonfinite_dense_matrix():
    mcmc = SimpleNamespace(
        num_chains=1,
        last_state=SimpleNamespace(
            adapt_state=SimpleNamespace(
                inverse_mass_matrix={
                    ("x", "y"): np.asarray([[1.0, np.nan], [np.nan, 1.0]])
                },
                step_size=np.asarray(0.02),
            )
        ),
    )

    diagnostics = coremod.nuts_metric_diagnostics(mcmc)
    block = diagnostics["blocks"][0]
    assert block["n_nonfinite_eigenvalues"] == 2
    assert np.isinf(block["condition_number"])


def _install_mocked_numpyro_runtime(monkeypatch, q, *, host_basis_jax=None):
    """Install fast inference doubles and return their captured arguments."""
    captures = {"nuts": [], "init_values": [], "run": []}
    fsps_grid = coremod.FSPSTemplateGrid(
        wave=np.asarray(q.wave),
        templates=np.zeros((np.asarray(q.wave).size, 1)),
        template_meta=[{"norm": 1.0}],
        age_grid_gyr=(1.0,),
        logzsol_grid=(0.0,),
        host_basis_jax=host_basis_jax,
        t_obs_gyr=1.0,
    )
    monkeypatch.setattr(q, "_build_fsps_grid_for_fit", lambda **kwargs: fsps_grid)
    monkeypatch.setattr(q, "_consume_posterior_outputs", lambda **kwargs: None)

    def fake_init_to_value(*, values):
        captures["init_values"].append(dict(values))
        return object()

    class FakeNUTS:
        def __init__(self, model, **kwargs):
            captures["nuts"].append(kwargs)

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            self.num_chains = int(kwargs["num_chains"])
            self.last_state = None

        def run(self, key, **kwargs):
            captures["run"].append(kwargs)

        def print_summary(self):
            return None

        def get_samples(self, group_by_chain=False):
            return {}

        def get_extra_fields(self, group_by_chain=False):
            assert group_by_chain is True
            return {
                "num_steps": np.array([[3, 7]]),
                "accept_prob": np.array([[0.85, 0.95]]),
                "diverging": np.array([[False, False]]),
                "potential_energy": np.array([[1.0, 1.2]]),
                "energy": np.array([[1.1, 1.4]]),
            }

    class FakePredictive:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(coremod, "init_to_value", fake_init_to_value)
    monkeypatch.setattr(coremod, "NUTS", FakeNUTS)
    monkeypatch.setattr(coremod, "MCMC", FakeMCMC)
    monkeypatch.setattr(coremod, "Predictive", FakePredictive)
    return captures


def test_numpyro_full_dense_precedes_blocks_and_uses_warmup_depth_tuple(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave, q.flux, q.err = lam, flux, err
    q.fe_uv_wave = q.fe_op_wave = np.array([lam[0], lam[-1]])
    q.fe_uv_flux = q.fe_op_flux = np.zeros(2)
    captures = _install_mocked_numpyro_runtime(monkeypatch, q)

    cfg = build_default_prior_config(flux).to_mapping()
    cfg.pop("line", None)
    cfg["line_block_dense_mass"] = True

    q.run_fsps_numpyro_fit(
        num_warmup=2,
        num_samples=2,
        dense_mass=True,
        max_tree_depth=7,
        warmup_max_tree_depth=10,
        prior_config=cfg,
        use_lines=False,
        decompose_host=False,
        fit_fe=False,
        fit_bc=False,
    )

    nuts_kwargs = captures["nuts"][0]
    assert nuts_kwargs["dense_mass"] is True
    assert nuts_kwargs["max_tree_depth"] == (10, 7)
    assert nuts_kwargs["find_heuristic_step_size"] is True
    assert q.nuts_mass_matrix_structure is True
    assert captures["run"][0]["extra_fields"] == (
        "num_steps",
        "accept_prob",
        "potential_energy",
        "energy",
    )


def test_numpyro_delayed_host_warm_start_keeps_physical_shared_coordinates(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave, q.flux, q.err = lam, flux, err
    q.fe_uv_wave = q.fe_op_wave = np.array([lam[0], lam[-1]])
    q.fe_uv_flux = q.fe_op_flux = np.zeros(2)
    class _HostBasis:
        ssp_lg_age_gyr = np.log10(np.array([0.1, 1.0]))
        ssp_lgmet = np.array([-2.0, -1.0])

    captures = _install_mocked_numpyro_runtime(
        monkeypatch,
        q,
        host_basis_jax=_HostBasis(),
    )
    cfg = build_default_prior_config(flux).to_mapping()
    cfg.pop("line", None)
    cfg["host_sfh_model"] = "delayed"
    cfg["standardize_active_priors"] = True

    q.run_fsps_numpyro_fit(
        num_warmup=2,
        num_samples=2,
        dense_mass=True,
        prior_config=cfg,
        use_lines=False,
        decompose_host=True,
        fit_fe=False,
        fit_bc=False,
    )

    values = captures["init_values"][0]
    physical_shared = {
        "log_stellar_mass",
        "log_sfh_age_gyr",
        "log_sfh_tau_over_age",
        "gal_lgmet",
        "log_gal_lgmet_scatter",
    }
    assert physical_shared <= set(values)
    assert not any(f"{name}_std" in values for name in physical_shared)
    assert values["log_host_aperture_scale_std"] == 0.0
    assert np.log(0.01) < values["log_sfh_age_gyr"] < np.log(1.0)
    assert -2.0 < values["gal_lgmet"] < -1.0


def test_delayed_host_init_respects_high_redshift_age_and_ssp_metallicity_support():
    class _HostBasis:
        ssp_lg_age_gyr = np.log10(np.array([0.01, 0.1, 1.0]))
        ssp_lgmet = np.array([-4.35, -2.5, -1.35])

    grid = coremod.FSPSTemplateGrid(
        wave=np.linspace(3000.0, 5000.0, 8),
        templates=np.ones((8, 2)),
        template_meta=[{"tage_gyr": 0.1}, {"tage_gyr": 1.0}],
        age_grid_gyr=(0.1, 1.0),
        logzsol_grid=(-1.0, 0.0),
        host_basis_jax=_HostBasis(),
        t_obs_gyr=1.2,
    )
    cfg = build_default_prior_config(np.ones(8)).to_mapping()

    values = coremod._build_delayed_host_init_values(cfg, grid)

    assert np.log(0.01) < values["log_sfh_age_gyr"] < np.log(1.2)
    assert -4.35 < values["gal_lgmet"] < -1.35
    assert "log_sfh_tau_over_age" in values
    age_gyr = np.exp(values["log_sfh_age_gyr"])
    assert np.log(0.03 / age_gyr) < values["log_sfh_tau_over_age"]
    assert values["log_sfh_tau_over_age"] < np.log(30.0 / age_gyr)


def test_flexible_host_map_init_is_converted_to_exact_nuts_noise():
    class _Grid:
        templates = np.ones((8, 3))
        template_meta = [
            {"tage_gyr": 0.1},
            {"tage_gyr": 1.0},
            {"tage_gyr": 10.0},
        ]

    cfg = {
        "host_sfh_model": "flexible",
        "standardize_active_priors": True,
        "host_template_age_prior": {"enabled": False},
        "raw_w": {"dist": "Normal", "loc": -0.4, "scale": 1.0},
        "tau_host": dist.HalfNormal(1.0),
    }
    tau_std = np.array(0.25)
    tau = np.asarray(
        modelmod._standardized_prior_value(
            cfg["tau_host"], coremod.jnp.asarray(tau_std)
        )
    )
    expected_noise = np.array([-1.0, 0.5, 1.25])
    raw = -0.4 + tau * expected_noise

    converted = coremod._convert_flexible_host_init_for_nuts(
        {
            "tau_host_std": tau_std,
            "fsps_weights_raw": raw,
        },
        _Grid(),
        cfg,
    )

    assert "fsps_weights_raw" not in converted
    np.testing.assert_allclose(
        converted["fsps_weights_raw_decentered"], expected_noise
    )
    np.testing.assert_allclose(converted["tau_host_std"], tau_std)


def test_flexible_host_nuts_reparam_keeps_physical_samples_for_predictive():
    def model():
        tau = numpyro.sample("tau_host", dist.HalfNormal(1.0))
        numpyro.sample(
            "fsps_weights_raw",
            dist.Normal(np.array([-0.4, 0.2]), tau),
        )

    config = coremod._numpyro_geometry_reparam_config(
        {"host_sfh_model": "flexible"},
        decompose_host=True,
    )
    nuts_model = coremod.reparam(model, config=config)
    mcmc = coremod.MCMC(
        coremod.NUTS(nuts_model),
        num_warmup=5,
        num_samples=5,
        progress_bar=False,
    )
    mcmc.run(coremod.jax.random.PRNGKey(10))
    samples = mcmc.get_samples()

    assert "fsps_weights_raw" in samples
    assert "fsps_weights_raw_decentered" in samples
    pred = coremod.Predictive(
        model,
        posterior_samples=samples,
        return_sites=["fsps_weights_raw"],
    )
    pred_out = pred(coremod.jax.random.PRNGKey(11))
    np.testing.assert_allclose(
        pred_out["fsps_weights_raw"], samples["fsps_weights_raw"]
    )


def test_fit_dispatch_nuts_dereddens_psf_phot_when_enabled(monkeypatch):
    lam, flux, err = _make_wide_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1, ra=150.0, dec=2.0)

    called = {'nuts': 0, 'kwargs': None}

    def _stub_nuts(**kwargs):
        called['nuts'] += 1
        called['kwargs'] = kwargs

    def _stub_deredden(lam_in, flux_in, err_in, ra_in, dec_in):
        q.ebv_mw = 0.15
        q.flux = flux_in
        q.err = err_in
        return q.flux

    monkeypatch.setattr(q, 'run_fsps_numpyro_fit', _stub_nuts)
    monkeypatch.setattr(q, '_de_redden', _stub_deredden)

    q.config.inference.method = 'nuts'
    q.config.observation.apply_mw_deredden = True
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.config.psf_photometry = coremod.PSFPhotometryData(
        magnitudes=np.array([19.8, 19.6]),
        magnitude_errors=np.array([0.05, 0.06]),
        filter_names=["g", "r"],
    )
    q.fit()

    assert called['nuts'] == 1
    assert called['kwargs']['use_psf_phot'] is True
    assert np.all(called['kwargs']['psf_mags'] < np.array([19.8, 19.6]))
    assert np.allclose(q.psf_mags_raw, np.array([19.8, 19.6]))
    assert np.allclose(q.psf_mags_dered, called['kwargs']['psf_mags'])


def test_fit_dispatch_optax(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax': 0, 'kwargs': None}

    def _stub_optax(**kwargs):
        called['optax'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.config.inference.method = 'optax'
    q.config.inference.random_seed = 73
    q.config.inference.plot_init = True
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['optax'] == 1
    assert called['kwargs']['plot_init'] is True
    assert called['kwargs']['random_seed'] == 73


def test_fit_dispatch_optax_accepts_output_plot_init(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {'kwargs': None}

    def _stub_optax(**kwargs):
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.config.inference.method = 'optax'
    q.config.inference.plot_init = False
    q.config.output.plot_init = True
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['kwargs']['plot_init'] is True


def test_plot_initialization_uses_packaged_matplotlib_style(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    pred_out = {
        "model": np.asarray([flux]),
        "gal_model": np.asarray([0.2 * flux]),
        "f_pl_model": np.asarray([0.8 * flux]),
        "line_model": np.zeros((1, flux.size)),
        "continuum_model": np.asarray([flux]),
    }
    entered = {"style": 0}

    @contextmanager
    def _style_context():
        entered["style"] += 1
        yield

    monkeypatch.setattr("jaxsedfit.mplstyle.use_style", _style_context)
    monkeypatch.setattr("jaxqsofit.plotting.plt.show", lambda: None)

    q._plot_initialization(
        lam,
        flux,
        err,
        pred_out,
        {"x": 1.0},
        stage_name="test init",
        attr_prefix="init_test",
        model_label="init model",
    )

    assert entered["style"] == 1
    assert hasattr(q, "init_test_model")


def test_fit_builds_default_priors_from_rest_frame_flux(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    z = 2.0
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=z)

    called = {'kwargs': None}

    def _stub_optax(**kwargs):
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.config.inference.method = 'optax'
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = None
    q.fit()

    prior_config = called['kwargs']['prior_config']
    expected_rest_fscale = np.nanmedian(np.abs(flux * (1.0 + z)))
    observed_fscale = np.nanmedian(np.abs(flux))

    assert np.isclose(prior_config["cont_norm"].loc, np.log(expected_rest_fscale))
    assert not np.isclose(prior_config["cont_norm"].loc, np.log(observed_fscale))
    assert np.allclose(q.flux, flux * (1.0 + z))


def test_fit_applies_true_means_keep_mask_before_preprocessing_and_keeps_zero_flux(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    mask = np.ones(lam.size, dtype=bool)
    mask[1] = False
    flux[2] = 0.0
    flux[3] = np.nan
    err[4] = 0.0
    lam[5] = np.nan
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, mask=mask, z=0.0)

    monkeypatch.setattr(q, "run_fsps_optax_fit", lambda **kwargs: None)
    q.config.inference.method = "optax"
    q.config.observation.apply_mw_deredden = False
    q.config.preprocessing.mask_lya_forest = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.fit()

    expected_keep = mask & np.isfinite(lam) & (lam > 0.0) & np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    assert np.array_equal(q.lam, lam[expected_keep])
    assert np.array_equal(q.flux, flux[expected_keep])
    assert q.flux[np.where(q.lam == lam[2])[0][0]] == 0.0


def test_fit_bal_appends_builtin_bal_components(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax': 0, 'kwargs': None}

    def _stub_optax(**kwargs):
        called['optax'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.config.inference.method = 'optax'
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.bal.enabled = True
    q.config.bal.tau_scale = 0.5
    q.config.bal.covering_loc = 0.35
    q.config.bal.covering_scale = 0.11
    q.config.bal.covering_high = 0.80
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['optax'] == 1
    components = called['kwargs']['custom_components']
    names = [comp.name for comp in components]
    assert names == ["bal_nv", "bal_siiv", "bal_civ"]
    for comp in components:
        assert np.isclose(comp.parameter_priors["tau_peak"].scale, 0.5)
        covering_cfg = comp.parameter_priors["covering"]
        assert np.isclose(covering_cfg.base_dist.loc, 0.35)
        assert np.isclose(covering_cfg.base_dist.scale, 0.11)
        assert np.isclose(covering_cfg.high, 0.80)


def test_fit_dispatch_optax_nuts(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax_nuts': 0, 'kwargs': None}

    def _stub_optax_nuts(**kwargs):
        called['optax_nuts'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_nuts_fit', _stub_optax_nuts)

    q.config.inference.method = 'optax+nuts'
    q.config.inference.random_seed = 41
    q.config.inference.plot_init = True
    q.config.inference.dense_mass = False
    q.config.inference.max_tree_depth = 7
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['optax_nuts'] == 1
    assert called['kwargs']['plot_init'] is True
    assert called['kwargs']['dense_mass'] is False
    assert called['kwargs']['max_tree_depth'] == 7
    assert called['kwargs']['random_seed'] == 41


def test_optax_warm_start_subsets_psf_filters_for_stage1(monkeypatch):
    lam = np.linspace(2500.0, 10000.0, 256)
    flux = 40.0 + 0.0015 * (lam - 6000.0)
    err = np.full_like(flux, 0.4)
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.flux = flux
    q.err = err
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q.verbose = False

    fsps_grid = coremod.FSPSTemplateGrid(
        wave=lam,
        templates=np.zeros((lam.size, 1)),
        template_meta=[{"norm": 1.0}],
        age_grid_gyr=(1.0,),
        logzsol_grid=(0.0,),
        host_basis_jax=None,
        t_obs_gyr=None,
    )
    monkeypatch.setattr(q, "_build_fsps_grid_for_fit", lambda **kwargs: fsps_grid)
    monkeypatch.setattr(q, "_consume_posterior_outputs", lambda **kwargs: None)

    svi_calls = []
    svi_keys = []

    class FakeSVIResult:
        losses = np.array([0.0])
        params = {}
        state = object()

    class FakeSVI:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, key, steps, **kwargs):
            svi_keys.append(np.asarray(key))
            svi_calls.append(kwargs)
            return FakeSVIResult()

    class FakeAutoDelta:
        def __init__(self, *args, **kwargs):
            pass

        def median(self, params):
            return {"PL_norm": np.array(1.0), "PL_slope": np.array(-1.5)}

    class FakePredictive:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(coremod, "SVI", FakeSVI)
    monkeypatch.setattr(coremod, "AutoDelta", FakeAutoDelta)
    monkeypatch.setattr(coremod, "Predictive", FakePredictive)

    psf_filter_curves = {
        "bands": ("g", "r"),
        "trans": np.vstack([
            np.linspace(0.0, 1.0, lam.size),
            np.linspace(1.0, 0.0, lam.size),
        ]),
        "coverage": np.array([1.0, 1.0]),
    }

    q.run_fsps_optax_fit(
        num_steps=300,
        use_lines=False,
        decompose_host=False,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        prior_config=build_default_prior_config(flux),
        psf_mags=np.array([19.8, 19.6]),
        psf_mag_errs=np.array([0.05, 0.06]),
        psf_filter_curves=psf_filter_curves,
        use_psf_phot=True,
        random_seed=19,
    )

    assert len(svi_calls) == 2
    stage1_keep = q.init_stage1_keep_mask
    assert not stage1_keep[np.argmin(np.abs(q.wave - 2798.75))]
    assert svi_calls[0]["wave"].shape[0] == int(np.sum(stage1_keep))
    assert svi_calls[0]["psf_filter_curves"]["trans"].shape == (2, int(np.sum(stage1_keep)))
    assert svi_calls[1]["wave"].shape[0] == q.wave.size
    assert svi_calls[1]["psf_filter_curves"]["trans"].shape == (2, q.wave.size)
    assert svi_calls[1]["psf_filter_curves"] is psf_filter_curves
    expected_key = np.asarray(coremod.jax.random.PRNGKey(19))
    assert all(np.array_equal(key, expected_key) for key in svi_keys)


def test_optax_stage2_initializes_reparameterized_line_sites_at_defaults(monkeypatch):
    lam, flux, err = _make_wide_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.flux = flux
    q.err = err
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q.verbose = False

    fsps_grid = coremod.FSPSTemplateGrid(
        wave=lam,
        templates=np.zeros((lam.size, 1)),
        template_meta=[{"norm": 1.0}],
        age_grid_gyr=(1.0,),
        logzsol_grid=(0.0,),
        host_basis_jax=None,
        t_obs_gyr=None,
    )
    monkeypatch.setattr(q, "_build_fsps_grid_for_fit", lambda **kwargs: fsps_grid)
    monkeypatch.setattr(q, "_consume_posterior_outputs", lambda **kwargs: None)

    init_values_seen = []

    def fake_init_to_value(*, values):
        init_values_seen.append(dict(values))
        return object()

    class FakeSVIResult:
        losses = np.array([0.0])
        params = {}
        state = object()

    class FakeSVI:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return FakeSVIResult()

    class FakeAutoDelta:
        def __init__(self, *args, **kwargs):
            pass

        def median(self, params):
            return {"PL_norm": np.array(1.0), "PL_slope": np.array(-1.5)}

    class FakePredictive:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(coremod, "init_to_value", fake_init_to_value)
    monkeypatch.setattr(coremod, "SVI", FakeSVI)
    monkeypatch.setattr(coremod, "AutoDelta", FakeAutoDelta)
    monkeypatch.setattr(coremod, "Predictive", FakePredictive)

    q.run_fsps_optax_fit(
        num_steps=300,
        use_lines=True,
        decompose_host=False,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        prior_config=build_default_prior_config(flux),
    )

    assert len(init_values_seen) == 2
    stage2_values = init_values_seen[1]
    assert np.allclose(stage2_values["line_dmu_independent_group_std"], 0.0)
    assert np.allclose(stage2_values["line_log_fwhm_delta_group_std"], 0.0)
    amp_sites = [
        value
        for key, value in stage2_values.items()
        if key.startswith("line_amp_")
        and key != "line_amp_group"
        and not key.endswith("_std")
    ]
    assert amp_sites
    assert all(np.all(value > 0.0) for value in amp_sites)
    assert np.isclose(stage2_values["line_log_broad_fwhm_std"], 0.0)
    assert "line_log_narrow_fwhm_std" not in stage2_values
    assert np.isclose(stage2_values["line_OIII_wing_log_fwhm_std"], 0.0)


def test_optax_reddening_warm_start_transfers_apparent_power_law_sites(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.flux = flux
    q.err = err
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.zeros(2)
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.zeros(2)
    q.verbose = False

    fsps_grid = coremod.FSPSTemplateGrid(
        wave=lam,
        templates=np.zeros((lam.size, 1)),
        template_meta=[{"norm": 1.0}],
        age_grid_gyr=(1.0,),
        logzsol_grid=(0.0,),
        host_basis_jax=None,
        t_obs_gyr=None,
    )
    monkeypatch.setattr(q, "_build_fsps_grid_for_fit", lambda **kwargs: fsps_grid)
    monkeypatch.setattr(q, "_consume_posterior_outputs", lambda **kwargs: None)

    svi_calls = []
    init_values_seen = []

    def fake_init_to_value(*, values):
        init_values_seen.append(dict(values))
        return object()

    class FakeSVIResult:
        losses = np.array([0.0])
        params = {}
        state = object()

    class FakeSVI:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            svi_calls.append(kwargs)
            return FakeSVIResult()

    class FakeAutoDelta:
        def __init__(self, *args, **kwargs):
            pass

        def median(self, params):
            return {
                "PL_apparent_log_norm_std": np.array(0.2),
                "PL_apparent_slope_std": np.array(-0.1),
                "log_ebv": np.array(-4.0),
            }

    class FakePredictive:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(coremod, "init_to_value", fake_init_to_value)
    monkeypatch.setattr(coremod, "SVI", FakeSVI)
    monkeypatch.setattr(coremod, "AutoDelta", FakeAutoDelta)
    monkeypatch.setattr(coremod, "Predictive", FakePredictive)

    q.run_fsps_optax_fit(
        num_steps=300,
        use_lines=False,
        decompose_host=False,
        fit_fe=False,
        fit_bc=False,
        fit_poly=True,
        fit_reddening=True,
        prior_config=build_default_prior_config(flux),
    )

    assert svi_calls[0]["fit_reddening"] is True
    assert svi_calls[1]["fit_reddening"] is True
    assert "PL_apparent_log_norm_std" in init_values_seen[0]
    assert "PL_apparent_slope_std" in init_values_seen[0]
    assert "log_ebv" in init_values_seen[0]
    assert np.isclose(init_values_seen[1]["PL_apparent_log_norm_std"], 0.2)
    assert np.isclose(init_values_seen[1]["PL_apparent_slope_std"], -0.1)
    assert np.isclose(init_values_seen[1]["log_ebv"], -4.0)


def test_fit_materializes_default_pl_pivot_to_numeric(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    cfg = build_default_prior_config(flux).to_mapping()
    assert cfg["PL_pivot"] is None
    q.config.inference.method = 'optax'
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = cfg
    q.fit()

    pivot = q._fit_prior_config["PL_pivot"]
    assert isinstance(pivot, float)
    assert np.isfinite(pivot)
    poly_pivot = q._fit_prior_config["poly_pivot"]
    assert isinstance(poly_pivot, float)
    assert np.isfinite(poly_pivot)


def test_fit_preserves_explicit_pl_pivot_value(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    q.config.inference.method = 'optax'
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux, pl_pivot=3000.0)
    q.fit()

    assert q._fit_prior_config["PL_pivot"] == 3000.0
    assert isinstance(q._fit_prior_config["poly_pivot"], float)


def test_fit_materializes_missing_pl_pivot_key(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    cfg = build_default_prior_config(flux).to_mapping()
    cfg.pop("PL_pivot")
    q.config.inference.method = 'optax'
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = cfg
    q.fit()

    pivot = q._fit_prior_config["PL_pivot"]
    assert isinstance(pivot, float)
    assert np.isfinite(pivot)
    assert isinstance(q._fit_prior_config["poly_pivot"], float)


def test_inference_method_unknown_raises():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    q.config.inference.method = 'not-a-method'
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    with pytest.raises(ValueError, match='Unknown inference method'):
        q.fit()


def test_load_from_samples_roundtrip(tmp_path, monkeypatch):
    q, lam, flux, _err = _build_bundle_source(tmp_path, "unit_test_fit", decompose_host=False)
    saved_path = q.save_posterior_bundle()
    assert saved_path.endswith("unit_test_fit_samples.h5")

    called = {"plot_fig": 0, "plot_mcmc_diagnostics": 0}

    def _stub_plot_fig(self, **kwargs):
        called["plot_fig"] += 1

    def _stub_plot_mcmc_diagnostics(self, **kwargs):
        called["plot_mcmc_diagnostics"] += 1

    monkeypatch.setattr(JAXQSOFit, "plot_fig", _stub_plot_fig)
    monkeypatch.setattr(JAXQSOFit, "plot_mcmc_diagnostics", _stub_plot_mcmc_diagnostics)

    loaded = jaxqsofit.load_from_samples(
        filename="unit_test_fit",
        output_path=str(tmp_path),
    )

    assert isinstance(loaded, JAXQSOFit)
    assert loaded.filename == "unit_test_fit"
    assert loaded.output_path == str(tmp_path)
    assert np.allclose(loaded.lam_in, lam)
    assert np.allclose(loaded.flux_in, flux)
    assert hasattr(loaded, "model_total")
    assert loaded.model_total.shape == lam.shape
    assert set(loaded.numpyro_samples.keys()) == {
        "cont_norm",
        "log_frac_host",
        "PL_norm",
        "PL_slope",
        "frac_jitter",
        "add_jitter",
    }
    assert loaded.pred_out["fsps_weights"].shape == (3, 4)
    assert np.allclose(loaded.pred_out["fsps_weights"], 0.0)
    assert np.allclose(loaded._pred_host_draws, 0.0)
    assert np.allclose(loaded.host, 0.0)
    assert called["plot_fig"] == 1
    assert called["plot_mcmc_diagnostics"] == 1


def test_load_result_wraps_loaded_qsofit(tmp_path, monkeypatch):
    q, _lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_fit_result", decompose_host=False)
    saved_path = q.save_posterior_bundle()

    monkeypatch.setattr(JAXQSOFit, "plot_fig", lambda self, **kwargs: None)
    monkeypatch.setattr(JAXQSOFit, "plot_mcmc_diagnostics", lambda self, **kwargs: None)

    result = JAXQSOFit.load_result(filename="unit_test_fit_result", output_path=str(tmp_path))

    assert isinstance(result, FitResult)
    assert isinstance(result.fitter, JAXQSOFit)
    assert os.fspath(result.path) == os.fspath(saved_path)
    assert set(result.samples) == {
        "cont_norm",
        "log_frac_host",
        "PL_norm",
        "PL_slope",
        "frac_jitter",
        "add_jitter",
    }
    assert np.isclose(result.median["PL_norm"], 1.0)


def test_fit_result_save_and_plot_delegates(tmp_path, monkeypatch):
    q, _lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_result_methods", decompose_host=False)
    result = q._make_result(method="optax")
    calls = {"trace": None, "corner": None}

    def _trace(**kwargs):
        calls["trace"] = kwargs
        return "trace"

    def _corner(**kwargs):
        calls["corner"] = kwargs
        return "corner"

    monkeypatch.setattr(q, "plot_trace", _trace)
    monkeypatch.setattr(q, "plot_corner", _corner)

    saved_path = result.save(tmp_path, save_name="manual_result")

    assert os.fspath(result.path) == os.fspath(saved_path)
    assert os.path.exists(saved_path)
    assert result.plot_trace(show_plot=False) == "trace"
    assert result.plot_corner(show_plot=False) == "corner"
    assert calls["trace"]["show_plot"] is False
    assert calls["corner"]["show_plot"] is False


def test_fit_result_save_uses_captured_posterior_state(tmp_path):
    q, _lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_result_state", decompose_host=False)
    result = q._make_result(method="optax")

    q._posterior_state = type(q._posterior_state)(method="nuts")
    q.numpyro_samples = {"cont_norm": np.array([42.0])}

    saved_path = result.save(tmp_path, save_name="captured_state")

    with h5py.File(saved_path, "r") as h5f:
        np.testing.assert_allclose(h5f["samples"]["cont_norm"][()], [1.0, 1.1, 0.9])


def test_plot_spectrum_delegates_to_plot_fig(monkeypatch):
    q = object.__new__(JAXQSOFit)
    calls = {}

    def _plot_fig(self, **kwargs):
        calls["plot_fig"] = (self, kwargs)
        return "figure"

    monkeypatch.setattr(JAXQSOFit, "plot_fig", _plot_fig)

    assert q.plot_spectrum(show_plot=False, plot_legend=False) == "figure"
    assert calls["plot_fig"][0] is q
    assert calls["plot_fig"][1] == {"show_plot": False, "plot_legend": False}


def test_plot_trace_show_plot_false_skips_plt_show(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {"PL_slope": np.array([-1.5, -1.4, -1.6])}
    q.save_fig = False

    called = {"show": 0}

    def _stub_show():
        called["show"] += 1

    monkeypatch.setattr(plottingmod.plt, "show", _stub_show)

    fig = q.plot_trace(show_plot=False)

    assert fig is not None
    assert called["show"] == 0


def test_posterior_series_defaults_use_sampled_parameter_names():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {
        "Fe_uv_norm": np.array([1.0, 1.1]),
        "log_Fe_op_over_uv": np.array([0.0, 0.1]),
        "Balmer_vel": np.array([3000.0, 3200.0]),
        "Fe_op_norm": np.array([99.0, 99.0]),
        "Balmer_Te": np.array([15000.0, 15000.0]),
    }

    labels = [name for name, _ in plottingmod.posterior_series(q)]

    assert "log_Fe_op_over_uv" in labels
    assert "Balmer_vel" in labels
    assert "Fe_op_norm" not in labels
    assert "Balmer_Te" not in labels


def test_plot_trace_show_plot_true_calls_plt_show(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {"PL_slope": np.array([-1.5, -1.4, -1.6])}
    q.save_fig = False

    called = {"show": 0}

    def _stub_show():
        called["show"] += 1

    monkeypatch.setattr(plottingmod.plt, "show", _stub_show)

    fig = q.plot_trace(show_plot=True)

    assert fig is not None
    assert called["show"] == 1


def test_plot_corner_show_plot_false_skips_plt_show(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
        "cont_norm": np.array([1.0, 1.1, 0.9]),
    }
    q.save_fig = False

    called = {"show": 0}

    def _stub_show():
        called["show"] += 1

    monkeypatch.setattr(plottingmod.plt, "show", _stub_show)

    fig = q.plot_corner(show_plot=False)

    assert fig is not None
    assert called["show"] == 0


def test_plot_corner_show_plot_true_calls_plt_show(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
        "cont_norm": np.array([1.0, 1.1, 0.9]),
    }
    q.save_fig = False

    called = {"show": 0}

    def _stub_show():
        called["show"] += 1

    monkeypatch.setattr(plottingmod.plt, "show", _stub_show)

    fig = q.plot_corner(show_plot=True)

    assert fig is not None
    assert called["show"] == 1


def test_plot_corner_omits_constant_posterior_parameters(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
        "constant": np.ones(3),
    }
    q.save_fig = False

    called = {}

    def _stub_corner(data, **kwargs):
        called["data"] = data
        called.update(kwargs)
        fig, _ = plottingmod.plt.subplots()
        return fig

    monkeypatch.setattr("corner.corner", _stub_corner)

    with pytest.warns(RuntimeWarning, match="constant"):
        fig = q.plot_corner(param_names="all", show_plot=False)

    assert fig is not None
    assert called["data"].shape == (3, 1)
    assert called["labels"] == ["PL_slope"]


def test_plot_corner_reduces_tick_label_fontsize():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
        "cont_norm": np.array([1.0, 1.1, 0.9]),
    }
    q.save_fig = False

    fig = q.plot_corner(show_plot=False)

    assert fig is not None
    tick_sizes = [
        tick.get_fontsize()
        for ax in fig.axes
        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        if tick.get_visible()
    ]
    assert tick_sizes
    assert all(size == 8 for size in tick_sizes)


def test_plot_corner_uses_light_curve_full_corner_rendering(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.numpyro_samples = {
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
        "cont_norm": np.array([1.0, 1.1, 0.9]),
    }
    q.save_fig = False

    called = {}

    def _stub_corner(*args, **kwargs):
        called.update(kwargs)
        fig, _ = plottingmod.plt.subplots()
        return fig

    monkeypatch.setattr("corner.corner", _stub_corner)

    fig = q.plot_corner(show_plot=False)

    assert fig is not None
    assert called["plot_datapoints"] is False
    assert called["plot_contours"] is True
    assert called["hist2d_kwargs"] == {"bins": 15, "levels": [0.393, 0.865, 0.989]}
    assert called["fill_contours"] is False
    assert called["no_fill_contours"] is True
    assert called["smooth"] == 0.8
    assert called["smooth1d"] == 0.8
    assert called["max_n_ticks"] == 3
    assert called["quiet"] is True
    assert called["labelpad"] == 0.3
    assert called["label_kwargs"] == {"fontsize": 9}
    assert called["title_kwargs"] == {"fontsize": 9}


def test_plot_mcmc_diagnostics_forwards_show_plot(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {"trace": None, "corner": None}

    def _stub_trace(self, **kwargs):
        called["trace"] = kwargs
        return None

    def _stub_corner(self, **kwargs):
        called["corner"] = kwargs
        return None

    monkeypatch.setattr(JAXQSOFit, "plot_trace", _stub_trace)
    monkeypatch.setattr(JAXQSOFit, "plot_corner", _stub_corner)

    q.plot_mcmc_diagnostics(show_plot=False)

    assert called["trace"] is not None
    assert called["corner"] is not None
    assert called["trace"]["show_plot"] is False
    assert called["corner"]["show_plot"] is False


def test_load_from_samples_roundtrip_without_filename(tmp_path, monkeypatch):
    q, _lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_fit_auto", decompose_host=False)
    q.save_posterior_bundle()

    monkeypatch.setattr(JAXQSOFit, "plot_fig", lambda self, **kwargs: None)
    monkeypatch.setattr(JAXQSOFit, "plot_mcmc_diagnostics", lambda self, **kwargs: None)

    loaded = jaxqsofit.load_from_samples(output_path=str(tmp_path))

    assert isinstance(loaded, JAXQSOFit)
    assert loaded.filename == "unit_test_fit_auto"
    assert loaded.output_path == str(tmp_path)


def test_load_from_samples_roundtrip_with_host_enabled(tmp_path, monkeypatch):
    q, lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_fit_host", decompose_host=True)
    q.save_posterior_bundle()

    def _stub_template_grid(**kwargs):
        wave = np.asarray(kwargs["wave_out"], dtype=float)
        grid = type("Grid", (), {})()
        grid.wave = wave
        grid.templates = np.column_stack([
            np.ones(wave.size, dtype=float),
            np.ones(wave.size, dtype=float) * 2.0,
            np.ones(wave.size, dtype=float) * 3.0,
            np.ones(wave.size, dtype=float) * 4.0,
        ])
        grid.template_meta = [
            {"tage_gyr": 0.1, "logzsol": -0.5, "norm": 1.0, "dsps_lgmet": -1.0, "dsps_lg_age_gyr": -1.0},
            {"tage_gyr": 1.0, "logzsol": -0.5, "norm": 1.0, "dsps_lgmet": -1.0, "dsps_lg_age_gyr": 0.0},
            {"tage_gyr": 0.1, "logzsol": 0.0, "norm": 1.0, "dsps_lgmet": 0.0, "dsps_lg_age_gyr": -1.0},
            {"tage_gyr": 1.0, "logzsol": 0.0, "norm": 1.0, "dsps_lgmet": 0.0, "dsps_lg_age_gyr": 0.0},
        ]
        grid.age_grid_gyr = np.array([0.1, 1.0], dtype=float)
        grid.logzsol_grid = np.array([-0.5, 0.0], dtype=float)
        return grid

    monkeypatch.setattr(coremod, "build_fsps_template_grid", _stub_template_grid)
    monkeypatch.setattr(modelmod, "build_fsps_template_grid", _stub_template_grid)
    monkeypatch.setattr(JAXQSOFit, "plot_fig", lambda self, **kwargs: None)
    monkeypatch.setattr(JAXQSOFit, "plot_mcmc_diagnostics", lambda self, **kwargs: None)

    loaded = jaxqsofit.load_from_samples(
        filename="unit_test_fit_host",
        output_path=str(tmp_path),
    )

    recon = loaded.reconstruct_posterior_spectrum(n_draws=2)

    assert isinstance(loaded, JAXQSOFit)
    assert loaded.pred_out["fsps_weights"].shape == (3, 4)
    assert np.all(np.isfinite(loaded.pred_out["fsps_weights"]))
    assert loaded.host.shape == lam.shape
    assert recon["draws"]["host"].shape == (2, recon["wave"].size)


def test_save_posterior_bundle_excludes_figures_transient_and_duplicate_caches(tmp_path):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        filename="unit_test_prune",
        output_path=str(tmp_path),
    )

    q.wave = lam
    q.flux = flux
    q.err = err
    q.numpyro_samples = {"PL_norm": np.array([1.0, 0.9])}
    q._fit_prior_config = build_default_prior_config(flux)
    q._fit_fit_lines = False
    q._fit_decompose_host = False
    q._fit_fit_pl = True
    q._fit_fit_fe = False
    q._fit_fit_bc = False
    q._fit_fit_poly = False
    q._fit_fit_poly_order = 2
    q._fit_fit_reddening = False
    q._fit_fsps_age_grid = (0.1, 1.0)
    q._fit_fsps_logzsol_grid = (-0.5, 0.0)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q._pred_total_draws = np.ones((2, lam.size))
    q._pred_line_draws = np.ones((2, lam.size))
    q.numpyro_mcmc = object()
    q.svi = object()
    q.svi_state = object()
    q.fig = "fake-figure-state"
    q.trace_fig = "fake-trace-state"
    q.corner_fig = "fake-corner-state"
    q.fe_uv = np.ones((12, 2))
    q.fe_op = np.ones((12, 2))
    q.fsps_grid = type(
        "Grid",
        (),
        {
            "templates": np.ones((lam.size, 8)),
            "age_grid_gyr": np.array([0.1, 1.0]),
            "logzsol_grid": np.array([-0.5, 0.0]),
        },
    )()
    saved_path = q.save_posterior_bundle()
    with h5py.File(saved_path, "r") as h5f:
        assert "samples" in h5f
        assert "meta" in h5f
        assert "state" not in h5f
        assert "PL_norm" in h5f["samples"]
        assert "lam_in" in h5f["meta"]
        assert "flux_in" in h5f["meta"]
        assert "wave" in h5f["meta"]
        assert "pred_out" not in h5f["meta"]
        assert "_pred_total_draws" not in h5f["meta"]
        assert "_pred_line_draws" not in h5f["meta"]


def test_save_posterior_bundle_normalizes_explicit_name_to_h5(tmp_path):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        filename="unit_test_named",
        output_path=str(tmp_path),
    )
    q.numpyro_samples = {"PL_norm": np.array([1.0, 0.9])}

    saved_path = q.save_posterior_bundle(save_name="manual_bundle")
    assert saved_path.endswith("manual_bundle.h5")
    assert os.path.exists(saved_path)


def test_normalize_posterior_bundle_name_h5_policy():
    assert JAXQSOFit._normalize_posterior_bundle_name("manual_bundle") == "manual_bundle.h5"
    assert JAXQSOFit._normalize_posterior_bundle_name("manual_bundle.h5") == "manual_bundle.h5"
    assert JAXQSOFit._normalize_posterior_bundle_name("legacy_only.pkl") == "legacy_only.pkl.h5"
    assert JAXQSOFit._normalize_posterior_bundle_name("legacy_only.pkl.gz") == "legacy_only.pkl.gz.h5"


def test_reconstruct_posterior_spectrum_delegates_to_model_helper(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    q.wave = lam
    q.flux = flux
    q.fsps_grid = type(
        "Grid",
        (),
        {
            "age_grid_gyr": np.array([0.1, 1.0]),
            "logzsol_grid": np.array([-0.5, 0.0]),
        },
    )()
    q.numpyro_samples = {
        "cont_norm": np.array([1.0, 1.1]),
        "log_frac_host": np.array([0.0, 0.1]),
    }
    q.pred_out = {"fsps_weights": np.ones((2, 4))}
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q._fit_prior_config = build_default_prior_config(flux)
    q._fit_fsps_age_grid = (0.1, 1.0)
    q._fit_fsps_logzsol_grid = (-0.5, 0.0)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q._fit_fit_poly = True
    q._fit_fit_poly_order = 3
    q._fit_fit_reddening = False
    q._fit_decompose_host = False
    q._posterior_hydrated = True

    captured = {}

    def _stub_reconstruct(**kwargs):
        captured.update(kwargs)
        return {
            "wave": np.asarray(kwargs["wave_out"]),
            "draws": {"continuum": np.ones((2, len(kwargs["wave_out"])))},
            "median": {"continuum": np.ones(len(kwargs["wave_out"]))},
        }

    monkeypatch.setattr(coremod, "reconstruct_posterior_components", _stub_reconstruct)

    out = q.reconstruct_posterior_spectrum(wave_min=2500.0, n_draws=2, return_components=False)
    result_pred = q._make_result(method="nuts").predict(wave_min=2500.0, n_draws=2, return_components=False)

    assert "wave_out" in captured
    assert captured["samples"] is q.numpyro_samples
    assert captured["pred_out"] is q.pred_out
    assert captured["age_grid_gyr"] == q._fit_fsps_age_grid
    assert captured["logzsol_grid"] == q._fit_fsps_logzsol_grid
    assert captured["dsps_ssp_fn"] == "fake_ssp.h5"
    assert captured["fit_poly"] is True
    assert captured["fit_poly_order"] == 3
    assert captured["fit_reddening"] is False
    assert captured["decompose_host"] is False
    assert captured["n_draws"] == 2
    assert captured["return_components"] is False
    assert np.isclose(np.min(captured["wave_out"]), 2500.0)
    assert isinstance(result_pred, PredictionResult)
    assert "continuum" in result_pred["draws"]
    assert "continuum" in result_pred.median["median"]
    dln = np.diff(np.log(captured["wave_out"]))
    assert np.allclose(dln, dln[0], rtol=1e-6)
    assert np.allclose(out["wave"], captured["wave_out"])


def test_reconstruct_posterior_spectrum_raises_on_fsps_weight_width_mismatch():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    q.wave = lam
    q.flux = flux
    q.numpyro_samples = {
        "cont_norm": np.array([1.0, 1.1]),
        "log_frac_host": np.array([0.0, 0.1]),
    }
    q.pred_out = {"fsps_weights": np.ones((2, 1))}
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q._fit_prior_config = build_default_prior_config(flux)
    q._fit_fsps_age_grid = (0.1, 1.0)
    q._fit_fsps_logzsol_grid = (-0.5, 0.0)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q._fit_fit_poly = False
    q._fit_fit_poly_order = 2
    q._fit_fit_reddening = False

    with pytest.raises(ValueError, match="fsps_weights.*width 4, got 1"):
        q.reconstruct_posterior_spectrum()


def test_load_from_samples_raises_on_missing_fsps_metadata(tmp_path, monkeypatch):
    q, _lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_missing_meta", decompose_host=False)
    saved_path = q.save_posterior_bundle()

    with h5py.File(saved_path, "a") as h5f:
        del h5f["meta"]["_fit_dsps_ssp_fn"]

    monkeypatch.setattr(JAXQSOFit, "plot_fig", lambda self, **kwargs: None)
    monkeypatch.setattr(JAXQSOFit, "plot_mcmc_diagnostics", lambda self, **kwargs: None)

    with pytest.raises(ValueError, match="missing required FSPS metadata.*_fit_dsps_ssp_fn"):
        jaxqsofit.load_from_samples(
            filename="unit_test_missing_meta",
            output_path=str(tmp_path),
        )


def test_load_from_samples_roundtrip_host_disabled_reconstructs_without_loading_fsps(tmp_path, monkeypatch):
    q, _lam, _flux, _err = _build_bundle_source(tmp_path, "unit_test_host_disabled_recon", decompose_host=False)
    q.save_posterior_bundle()

    def _boom(**kwargs):
        raise AssertionError("FSPS templates should not be loaded for host-disabled hydration or reconstruction")

    monkeypatch.setattr(coremod, "build_fsps_template_grid", _boom)
    monkeypatch.setattr(modelmod, "build_fsps_template_grid", _boom)
    monkeypatch.setattr(JAXQSOFit, "plot_fig", lambda self, **kwargs: None)
    monkeypatch.setattr(JAXQSOFit, "plot_mcmc_diagnostics", lambda self, **kwargs: None)

    loaded = jaxqsofit.load_from_samples(
        filename="unit_test_host_disabled_recon",
        output_path=str(tmp_path),
    )

    recon = loaded.reconstruct_posterior_spectrum(n_draws=3)

    assert loaded.pred_out["fsps_weights"].shape == (3, 4)
    assert np.allclose(loaded.pred_out["fsps_weights"], 0.0)
    assert np.allclose(loaded._pred_host_draws, 0.0)
    assert np.allclose(recon["draws"]["host"], 0.0)


def test_component_fraction_at_wave_reconstruct_uses_rebuilt_draws(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam

    def _stub_reconstruct(self, **kwargs):
        return {
            "wave": np.array([2500.0, 3000.0]),
            "draws": {
                "host": np.array([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]),
                "continuum": np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
            },
            "median": {},
        }

    monkeypatch.setattr(JAXQSOFit, "reconstruct_posterior_spectrum", _stub_reconstruct)

    frac, err_out = q.component_fraction_at_wave(
        component="host",
        wave0=2500.0,
        reference="continuum",
        reconstruct=True,
    )

    expected = np.array([0.2, 0.2, 0.2])
    p16, p50, p84 = np.percentile(expected, [16.0, 50.0, 84.0])
    assert np.isclose(frac, p50)
    assert np.isclose(err_out, 0.5 * (p84 - p16))
