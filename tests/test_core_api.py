import os
import h5py

import numpy as np
import pytest

import jaxqsofit
import jaxqsofit.core as coremod
from jaxqsofit import QSOFit, build_default_prior_config


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


def test_init_err_optional_defaults_to_small_value():
    lam, flux, _ = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, z=0.1)

    assert q.err_in.shape == flux.shape
    assert np.allclose(q.err_in, 1e-6)


def test_init_psf_defaults_band_labels():
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        psf_mags=np.array([20.0, 19.8, 19.6]),
        psf_mag_errs=np.array([0.1, 0.1, 0.1]),
    )

    assert q.psf_bands == ["u", "g", "r"]


def test_prepare_psf_photometry_masks_invalid_and_builds_transmissions():
    lam, flux, err = _make_wide_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

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
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)
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
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)
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
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    with pytest.raises(ValueError, match="fit\\(deredden=False\\)|valid sky coordinates"):
        q._validate_deredden_coordinates(ra=-999, dec=-999)


def test_build_fsps_grid_for_fit_skips_template_load_when_host_disabled(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

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

    assert grid.templates.shape == (lam.size, 1)
    assert np.allclose(grid.templates, 0.0)


def test_fit_dispatch_nuts(monkeypatch):
    lam, flux, err = _make_wide_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'nuts': 0, 'kwargs': None}

    def _stub_nuts(**kwargs):
        called['nuts'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_numpyro_fit', _stub_nuts)

    q.fit(
        deredden=False,
        fit_method='nuts',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
        psf_mags=np.array([19.8, 19.6]),
        psf_mag_errs=np.array([0.05, 0.06]),
        psf_bands=["g", "r"],
        use_psf_phot=True,
    )

    assert called['nuts'] == 1
    assert called['kwargs']['use_psf_phot'] is True
    assert called['kwargs']['psf_mags'].shape == (2,)
    assert called['kwargs']['psf_filter_curves']['trans'].shape == (2, q.lam.size)


def test_fit_dispatch_nuts_dereddens_psf_phot_when_enabled(monkeypatch):
    lam, flux, err = _make_wide_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1, ra=150.0, dec=2.0)

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

    q.fit(
        deredden=True,
        fit_method='nuts',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
        psf_mags=np.array([19.8, 19.6]),
        psf_mag_errs=np.array([0.05, 0.06]),
        psf_bands=["g", "r"],
        use_psf_phot=True,
    )

    assert called['nuts'] == 1
    assert called['kwargs']['use_psf_phot'] is True
    assert np.all(called['kwargs']['psf_mags'] < np.array([19.8, 19.6]))
    assert np.allclose(q.psf_mags_raw, np.array([19.8, 19.6]))
    assert np.allclose(q.psf_mags_dered, called['kwargs']['psf_mags'])


def test_fit_dispatch_optax(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax': 0}

    def _stub_optax(**kwargs):
        called['optax'] += 1

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.fit(
        deredden=False,
        fit_method='optax',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
    )

    assert called['optax'] == 1


def test_fit_bal_appends_builtin_bal_components(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax': 0, 'kwargs': None}

    def _stub_optax(**kwargs):
        called['optax'] += 1
        called['kwargs'] = kwargs

    monkeypatch.setattr(q, 'run_fsps_optax_fit', _stub_optax)

    q.fit(
        deredden=False,
        fit_method='optax',
        plot_fig=False,
        save_result=False,
        fit_bal=True,
        prior_config=build_default_prior_config(flux),
    )

    assert called['optax'] == 1
    names = [comp.name for comp in called['kwargs']['custom_components']]
    assert names == ["bal_nv", "bal_siiv", "bal_civ", "bal_ciii", "bal_mgii"]


def test_fit_dispatch_optax_nuts(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'optax_nuts': 0}

    def _stub_optax_nuts(**kwargs):
        called['optax_nuts'] += 1

    monkeypatch.setattr(q, 'run_fsps_optax_nuts_fit', _stub_optax_nuts)

    q.fit(
        deredden=False,
        fit_method='optax+nuts',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
    )

    assert called['optax_nuts'] == 1


def test_fit_materializes_default_pl_pivot_to_numeric(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    cfg = build_default_prior_config(flux)
    assert cfg["PL_pivot"] is None
    q.fit(
        deredden=False,
        fit_method='optax',
        plot_fig=False,
        save_result=False,
        prior_config=cfg,
    )

    pivot = q._fit_prior_config["PL_pivot"]
    assert isinstance(pivot, float)
    assert np.isfinite(pivot)


def test_fit_preserves_explicit_pl_pivot_value(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    q.fit(
        deredden=False,
        fit_method='optax',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux, pl_pivot=3000.0),
    )

    assert q._fit_prior_config["PL_pivot"] == 3000.0


def test_fit_materializes_missing_pl_pivot_key(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    monkeypatch.setattr(q, 'run_fsps_optax_fit', lambda **kwargs: None)

    cfg = build_default_prior_config(flux)
    cfg.pop("PL_pivot")
    q.fit(
        deredden=False,
        fit_method='optax',
        plot_fig=False,
        save_result=False,
        prior_config=cfg,
    )

    pivot = q._fit_prior_config["PL_pivot"]
    assert isinstance(pivot, float)
    assert np.isfinite(pivot)


def test_fit_method_unknown_raises():
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    with pytest.raises(ValueError, match='Unknown fit_method'):
        q.fit(
            deredden=False,
            fit_method='not-a-method',
            plot_fig=False,
            save_result=False,
            prior_config=build_default_prior_config(flux),
        )


def test_load_from_samples_roundtrip(tmp_path, monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        ra=150.0,
        dec=2.0,
        filename="unit_test_fit",
        output_path=str(tmp_path),
    )

    # Minimal fitted state needed for sample+meta reload/hydration.
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([3500.0, 7000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q._fit_prior_config = build_default_prior_config(flux)
    q._fit_fsps_age_grid = (0.1, 1.0)
    q._fit_fsps_logzsol_grid = (-0.5, 0.0)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q._fit_fit_lines = False
    q._fit_decompose_host = False
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
    q.save_fig = False

    saved_path = q.save_posterior_bundle()
    assert saved_path.endswith("unit_test_fit_samples.h5")

    called = {"plot_fig": 0, "plot_mcmc_diagnostics": 0}

    def _stub_plot_fig(self, **kwargs):
        called["plot_fig"] += 1

    def _stub_plot_mcmc_diagnostics(self, **kwargs):
        called["plot_mcmc_diagnostics"] += 1

    monkeypatch.setattr(QSOFit, "plot_fig", _stub_plot_fig)
    monkeypatch.setattr(QSOFit, "plot_mcmc_diagnostics", _stub_plot_mcmc_diagnostics)

    loaded = jaxqsofit.load_from_samples(
        filename="unit_test_fit",
        output_path=str(tmp_path),
    )

    assert isinstance(loaded, QSOFit)
    assert loaded.filename == "unit_test_fit"
    assert loaded.output_path == str(tmp_path)
    assert np.allclose(loaded.lam_in, lam)
    assert np.allclose(loaded.flux_in, flux)
    assert hasattr(loaded, "model_total")
    assert loaded.model_total.shape == lam.shape
    assert set(loaded.numpyro_samples.keys()) == {"cont_norm", "log_frac_host", "PL_norm", "PL_slope"}
    assert called["plot_fig"] == 1
    assert called["plot_mcmc_diagnostics"] == 1


def test_load_from_samples_roundtrip_without_filename(tmp_path, monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(
        lam=lam,
        flux=flux,
        err=err,
        z=0.1,
        ra=150.0,
        dec=2.0,
        filename="unit_test_fit_auto",
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
    q._fit_prior_config = build_default_prior_config(flux)
    q._fit_fsps_age_grid = (0.1, 1.0)
    q._fit_fsps_logzsol_grid = (-0.5, 0.0)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q._fit_fit_lines = False
    q._fit_decompose_host = False
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
    q.save_fig = False
    q.save_posterior_bundle()

    monkeypatch.setattr(QSOFit, "plot_fig", lambda self, **kwargs: None)
    monkeypatch.setattr(QSOFit, "plot_mcmc_diagnostics", lambda self, **kwargs: None)

    loaded = jaxqsofit.load_from_samples(output_path=str(tmp_path))

    assert isinstance(loaded, QSOFit)
    assert loaded.filename == "unit_test_fit_auto"
    assert loaded.output_path == str(tmp_path)


def test_save_posterior_bundle_excludes_figures_transient_and_duplicate_caches(tmp_path):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(
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
    q = QSOFit(
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
    assert QSOFit._normalize_posterior_bundle_name("manual_bundle") == "manual_bundle.h5"
    assert QSOFit._normalize_posterior_bundle_name("manual_bundle.h5") == "manual_bundle.h5"
    assert QSOFit._normalize_posterior_bundle_name("legacy_only.pkl") == "legacy_only.pkl.h5"
    assert QSOFit._normalize_posterior_bundle_name("legacy_only.pkl.gz") == "legacy_only.pkl.gz.h5"


def test_reconstruct_posterior_spectrum_delegates_to_model_helper(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

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
    q.pred_out = {"fsps_weights": np.ones((2, 2))}
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

    assert "wave_out" in captured
    assert captured["samples"] is q.numpyro_samples
    assert captured["pred_out"] is q.pred_out
    assert captured["age_grid_gyr"] == q._fit_fsps_age_grid
    assert captured["logzsol_grid"] == q._fit_fsps_logzsol_grid
    assert captured["dsps_ssp_fn"] == "fake_ssp.h5"
    assert captured["fit_poly"] is True
    assert captured["fit_poly_order"] == 3
    assert captured["fit_reddening"] is False
    assert captured["n_draws"] == 2
    assert captured["return_components"] is False
    assert np.isclose(np.min(captured["wave_out"]), 2500.0)
    assert np.allclose(out["wave"], captured["wave_out"])


def test_component_fraction_at_wave_reconstruct_uses_rebuilt_draws(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)
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

    monkeypatch.setattr(QSOFit, "reconstruct_posterior_spectrum", _stub_reconstruct)

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
