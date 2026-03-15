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


def test_init_err_optional_defaults_to_small_value():
    lam, flux, _ = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, z=0.1)

    assert q.err_in.shape == flux.shape
    assert np.allclose(q.err_in, 1e-6)


def test_fit_dispatch_nuts(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)

    called = {'nuts': 0}

    def _stub_nuts(**kwargs):
        called['nuts'] += 1

    monkeypatch.setattr(q, 'run_fsps_numpyro_fit', _stub_nuts)

    q.fit(
        deredden=False,
        fit_method='nuts',
        plot_fig=False,
        save_result=False,
        prior_config=build_default_prior_config(flux),
    )

    assert called['nuts'] == 1


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

    # Minimal fitted state needed for plotting/diagnostics reload.
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = flux * 0.98
    q.host = flux * 0.1
    q.f_pl_model = flux * 0.6
    q.f_fe_mgii_model = np.zeros_like(flux)
    q.f_fe_balmer_model = np.zeros_like(flux)
    q.f_bc_model = np.zeros_like(flux)
    q.f_line_model = np.zeros_like(flux)
    q.f_conti_model = q.host + q.f_pl_model
    q.pred_bands = {}
    q.decomposed = True
    q.line_result = np.array([], dtype=object)
    q.line_result_type = np.array([], dtype=object)
    q.line_result_name = np.array([], dtype=object)
    q.conti_result = np.array([], dtype=object)
    q.conti_result_type = np.array([], dtype=object)
    q.conti_result_name = np.array([], dtype=object)
    q.numpyro_samples = {
        "PL_norm": np.array([1.0, 1.1, 0.9]),
        "PL_slope": np.array([-1.5, -1.4, -1.6]),
    }
    q.save_fig = False

    saved_path = q.save_posterior_bundle()
    assert saved_path.endswith("unit_test_fit_samples.pkl")

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
    assert np.allclose(loaded.model_total, q.model_total)
    assert set(loaded.numpyro_samples.keys()) == {"PL_norm", "PL_slope"}
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
    q.model_total = flux * 0.98
    q.host = flux * 0.1
    q.f_pl_model = flux * 0.6
    q.f_fe_mgii_model = np.zeros_like(flux)
    q.f_fe_balmer_model = np.zeros_like(flux)
    q.f_bc_model = np.zeros_like(flux)
    q.f_line_model = np.zeros_like(flux)
    q.f_conti_model = q.host + q.f_pl_model
    q.pred_bands = {}
    q.decomposed = True
    q.line_result = np.array([], dtype=object)
    q.line_result_type = np.array([], dtype=object)
    q.line_result_name = np.array([], dtype=object)
    q.conti_result = np.array([], dtype=object)
    q.conti_result_type = np.array([], dtype=object)
    q.conti_result_name = np.array([], dtype=object)
    q.numpyro_samples = {
        "PL_norm": np.array([1.0, 1.1, 0.9]),
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
    q._fit_fit_poly_edge_flex = False

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
    assert captured["fit_poly_edge_flex"] is False
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
