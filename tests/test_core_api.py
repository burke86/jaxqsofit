import os
import h5py

import numpy as np
import pytest

import jaxqsofit
import jaxqsofit.core as coremod
import jaxqsofit.model as modelmod
from jaxqsofit import JAXQSOFit, build_default_prior_config
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
    assert called['kwargs']['psf_mags'].shape == (2,)
    assert called['kwargs']['psf_filter_curves']['trans'].shape == (2, q.lam.size)


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
    q.config.inference.plot_init = True
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['optax'] == 1
    assert called['kwargs']['plot_init'] is True


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

    assert np.isclose(prior_config["log_cont_norm"]["loc"], np.log(expected_rest_fscale))
    assert not np.isclose(prior_config["log_cont_norm"]["loc"], np.log(observed_fscale))
    assert np.allclose(q.flux, flux * (1.0 + z))


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
    q.config.continuum.fit_bal_absorption = True
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['optax'] == 1
    names = [comp.name for comp in called['kwargs']['custom_components']]
    assert names == ["bal_nv", "bal_siiv", "bal_civ"]


def test_fit_dispatch_optax_nuts(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax_nuts': 0, 'kwargs': None}

    def _stub_optax_nuts(**kwargs):
        called['optax_nuts'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_nuts_fit', _stub_optax_nuts)

    q.config.inference.method = 'optax+nuts'
    q.config.inference.plot_init = True
    q.config.observation.apply_mw_deredden = False
    q.config.output.plot_fig = False
    q.config.output.save_result = False
    q.config.prior_config = build_default_prior_config(flux)
    q.fit()

    assert called['optax_nuts'] == 1
    assert called['kwargs']['plot_init'] is True


def test_optax_warm_start_subsets_psf_filters_for_stage1(monkeypatch):
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

    svi_calls = []

    class FakeSVIResult:
        losses = np.array([0.0])
        params = {}
        state = object()

    class FakeSVI:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, key, steps, **kwargs):
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
    )

    assert len(svi_calls) == 2
    stage1_keep = q.init_stage1_keep_mask
    assert svi_calls[0]["wave"].shape[0] == int(np.sum(stage1_keep))
    assert svi_calls[0]["psf_filter_curves"]["trans"].shape == (2, int(np.sum(stage1_keep)))
    assert svi_calls[1]["wave"].shape[0] == q.wave.size
    assert svi_calls[1]["psf_filter_curves"]["trans"].shape == (2, q.wave.size)
    assert svi_calls[1]["psf_filter_curves"] is psf_filter_curves


def test_fit_materializes_default_pl_pivot_to_numeric(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    cfg = build_default_prior_config(flux)
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

    cfg = build_default_prior_config(flux)
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
    assert set(loaded.numpyro_samples.keys()) == {"cont_norm", "log_frac_host", "PL_norm", "PL_slope"}
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
    assert set(result.samples) == {"cont_norm", "log_frac_host", "PL_norm", "PL_slope"}
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

    monkeypatch.setattr(coremod.plt, "show", _stub_show)

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

    labels = [name for name, _ in q._posterior_series()]

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

    monkeypatch.setattr(coremod.plt, "show", _stub_show)

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

    monkeypatch.setattr(coremod.plt, "show", _stub_show)

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

    monkeypatch.setattr(coremod.plt, "show", _stub_show)

    fig = q.plot_corner(show_plot=True)

    assert fig is not None
    assert called["show"] == 1


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
        fig, _ = coremod.plt.subplots()
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
