import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist

import jaxqsofit.core as coremod
import jaxqsofit.model as modelmod
from jaxqsofit import (
    JAXQSOFit,
    make_custom_component,
    make_template_component,
)
from jaxqsofit.custom_components import (
    custom_component_param_site,
    custom_component_site_names,
    inject_default_custom_component_priors,
)


def _make_simple_spectrum():
    lam = np.linspace(2000.0, 4000.0, 64)
    flux = np.ones_like(lam)
    err = np.full_like(lam, 0.1)
    return lam, flux, err


def test_custom_component_prior_injection_and_site_names():
    comps = (
        make_template_component("Alt FeII", [2000.0, 2500.0, 3000.0], [0.0, 1.0, 0.0], fit_fwhm=True, fit_shift=True),
        make_custom_component(
            name="Blue Tilt",
            parameter_priors={
                "c0": {"dist": "Normal", "loc": 0.0, "scale": 0.0},
                "c1": {"dist": "Normal", "loc": 0.0, "scale": 0.0},
                "c2": {"dist": "Normal", "loc": 0.0, "scale": 0.0},
            },
            evaluate=lambda wave, params, metadata: params["c0"] + params["c1"] * 0.0 * wave + params["c2"] * 0.0 * wave,
        ),
    )
    cfg = inject_default_custom_component_priors({}, np.array([1.0, 2.0, 3.0]), comps)

    assert "custom_alt_feii_norm" in cfg
    assert "custom_alt_feii_fwhm" in cfg
    assert "custom_alt_feii_shift" in cfg
    assert "custom_blue_tilt_c0" in cfg
    assert "custom_blue_tilt_c1" in cfg
    assert "custom_blue_tilt_c2" in cfg
    assert custom_component_site_names(comps) == [
        "custom_alt_feii_model",
        "custom_blue_tilt_model",
    ]


def test_custom_component_accepts_numpyro_distribution_priors():
    comp = make_custom_component(
        name="Blue Tilt",
        parameter_priors={
            "c0": dist.Normal(0.0, 0.2),
            "amp": dist.LogNormal(np.log(0.1), 0.5),
        },
        evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["c0"] + params["amp"],
    )

    cfg = inject_default_custom_component_priors({}, np.array([1.0, 2.0, 3.0]), [comp])

    assert cfg["custom_blue_tilt_c0"] == {"dist": "Normal", "loc": 0.0, "scale": 0.2}
    assert cfg["custom_blue_tilt_amp"]["dist"] == "LogNormal"
    assert np.isclose(cfg["custom_blue_tilt_amp"]["loc"], np.log(0.1))
    assert cfg["custom_blue_tilt_amp"]["scale"] == 0.5


def test_custom_component_shared_parameter_site_injection():
    comp = make_custom_component(
        name="Shared BAL",
        parameter_priors={
            "v_out": {"dist": "TruncatedNormal", "loc": 6000.0, "scale": 2500.0, "low": 3000.0, "high": 12000.0},
            "tau_peak": {"dist": "HalfNormal", "scale": 1.0},
        },
        evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["tau_peak"] * 0.0,
        metadata={"shared_parameter_sites": {"v_out": "custom_bal_v_out"}},
    )

    cfg = inject_default_custom_component_priors({}, np.array([1.0, 2.0, 3.0]), [comp])

    assert custom_component_param_site(comp, "v_out") == "custom_bal_v_out"
    assert "custom_bal_v_out" in cfg
    assert "custom_shared_bal_v_out" not in cfg
    assert "custom_shared_bal_tau_peak" in cfg


def test_reconstruct_posterior_components_includes_custom_draws(monkeypatch):
    comps = (
        make_template_component("alt_fe", [2000.0, 2500.0, 3000.0], [0.0, 1.0, 0.0]),
        make_custom_component(
            name="blue_poly",
            parameter_priors={"c0": {"dist": "Normal", "loc": 0.0, "scale": 0.0}},
            evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["c0"],
        ),
    )

    class _Grid:
        templates = np.zeros((5, 1), dtype=float)
        age_grid_gyr = np.array([1.0], dtype=float)
        logzsol_grid = np.array([0.0], dtype=float)

    monkeypatch.setattr(modelmod, "build_fsps_template_grid", lambda **kwargs: _Grid())

    wave_out = np.linspace(2000.0, 3000.0, 5)
    samples = {
        "cont_norm": np.zeros(2),
        "log_frac_host": np.full(2, -100.0),
        "PL_slope": np.zeros(2),
        "custom_alt_fe_norm": np.array([1.0, 2.0]),
        "custom_blue_poly_c0": np.array([0.5, 1.0]),
    }

    out = modelmod.reconstruct_posterior_components(
        wave_out=wave_out,
        samples=samples,
        pred_out=None,
        age_grid_gyr=(1.0,),
        logzsol_grid=(0.0,),
        dsps_ssp_fn="fake.h5",
        prior_config={"PL_pivot": 2500.0},
        fit_poly=False,
        fit_reddening=False,
        fit_poly_order=0,
        fe_uv_wave=np.array([2000.0, 3000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 3000.0]),
        fe_op_flux=np.zeros(2),
        custom_components=comps,
        n_draws=2,
        return_components=True,
    )

    assert "alt_fe" in out["draws"]
    assert "blue_poly" in out["draws"]
    expected = out["draws"]["alt_fe"] + out["draws"]["blue_poly"]
    assert np.allclose(out["draws"]["continuum"], expected)


def test_reconstruct_posterior_components_host_disabled_uses_dummy_grid(monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("FSPS templates should not be loaded when decompose_host=False")

    monkeypatch.setattr(modelmod, "build_fsps_template_grid", _boom)

    wave_out = np.linspace(2000.0, 3000.0, 5)
    samples = {
        "cont_norm": np.array([1.0, 1.1]),
        "log_frac_host": np.array([0.0, 0.1]),
        "PL_norm": np.array([1.0, 1.0]),
        "PL_slope": np.array([0.0, 0.0]),
    }
    pred_out = {"fsps_weights": np.ones((2, 4))}

    out = modelmod.reconstruct_posterior_components(
        wave_out=wave_out,
        samples=samples,
        pred_out=pred_out,
        age_grid_gyr=(0.1, 1.0),
        logzsol_grid=(-0.5, 0.0),
        dsps_ssp_fn="fake.h5",
        prior_config={"PL_pivot": 2500.0},
        fit_poly=False,
        fit_reddening=False,
        fit_poly_order=0,
        fe_uv_wave=np.array([2000.0, 3000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 3000.0]),
        fe_op_flux=np.zeros(2),
        custom_components=(),
        n_draws=2,
        return_components=True,
        decompose_host=False,
    )

    assert np.allclose(out["draws"]["host"], 0.0)


def test_reconstruct_posterior_spectrum_passes_custom_components(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.flux = flux
    q.fsps_grid = type("Grid", (), {"age_grid_gyr": np.array([1.0]), "logzsol_grid": np.array([0.0])})()
    q.numpyro_samples = {"cont_norm": np.array([1.0])}
    q.pred_out = {"fsps_weights": np.ones((1, 1))}
    q.fe_uv_wave = np.array([2000.0, 4000.0])
    q.fe_uv_flux = np.array([0.0, 0.0])
    q.fe_op_wave = np.array([2000.0, 4000.0])
    q.fe_op_flux = np.array([0.0, 0.0])
    q._fit_prior_config = {}
    q._fit_fsps_age_grid = (1.0,)
    q._fit_fsps_logzsol_grid = (0.0,)
    q._fit_dsps_ssp_fn = "fake_ssp.h5"
    q._fit_fit_poly = False
    q._fit_fit_reddening = False
    q._fit_fit_poly_order = 0
    q._fit_custom_components = (
        make_custom_component(
            name="blue_poly",
            parameter_priors={
                "c0": {"dist": "Normal", "loc": 0.0, "scale": 1.0},
                "c1": {"dist": "Normal", "loc": 0.0, "scale": 1.0},
            },
            evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["c0"] + params["c1"],
        ),
    )

    captured = {}

    def _stub_reconstruct(**kwargs):
        captured.update(kwargs)
        return {
            "wave": np.asarray(kwargs["wave_out"]),
            "draws": {"continuum": np.ones((1, len(kwargs["wave_out"])))},
            "median": {"continuum": np.ones(len(kwargs["wave_out"]))},
        }

    monkeypatch.setattr(coremod, "reconstruct_posterior_components", _stub_reconstruct)

    q.reconstruct_posterior_spectrum(n_draws=1)

    assert captured["custom_components"] == q._fit_custom_components


def test_make_custom_component_supports_general_function():
    def _eval(wave, params, metadata):
        return params["amp"] * np.exp(-0.5 * ((wave - metadata["mu"]) / metadata["sigma"]) ** 2)

    comp = make_custom_component(
        name="gauss_bump",
        parameter_priors={"amp": {"dist": "LogNormal", "loc": 0.0, "scale": 0.5}},
        evaluate=_eval,
        metadata={"mu": 2500.0, "sigma": 100.0},
    )

    assert comp.site_name("amp") == "custom_gauss_bump_amp"


def test_plot_fig_includes_custom_component_trace():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.full_like(lam, 1.2)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {"smc_like_reddened_pl": np.full_like(lam, 0.8)}
    q.pred_bands = {
        "total_model": (np.full_like(lam, 1.1), np.full_like(lam, 1.3)),
        "host": (np.zeros_like(lam), np.zeros_like(lam)),
        "PL": (np.zeros_like(lam), np.zeros_like(lam)),
        "FeII": (np.zeros_like(lam), np.zeros_like(lam)),
        "Balmer_cont": (np.zeros_like(lam), np.zeros_like(lam)),
        "lines": (np.zeros_like(lam), np.zeros_like(lam)),
        "smc_like_reddened_pl": (np.full_like(lam, 0.7), np.full_like(lam, 0.9)),
    }
    q.scale_psf = 1.0
    q.save_fig = False
    q.custom_line_components = {}
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=True, plot_1sigma=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "smc like reddened pl" in labels
    plt.close(fig)


def test_plot_fig_tolerates_missing_prediction_bands():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.full_like(lam, 1.2)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {}
    q.pred_bands = None
    q.scale_psf = 1.0
    q.save_fig = False
    q.custom_line_components = {}
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=False, plot_1sigma=True)

    assert q.fig is not None
    plt.close(q.fig)


def test_plot_fig_scales_broad_component_overlay_to_fitted_broad_model():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    lnwave = np.log(lam)
    raw_broad = 5.0 * np.exp(-0.5 * ((lnwave - np.log(3000.0)) / 0.08) ** 2)
    q.line_broad = 0.25 * raw_broad
    q.line_narrow = np.zeros_like(lam)
    q.model_total = q.line_broad
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = q.line_broad
    q.custom_components = {}
    q.pred_bands = None
    q.scale_psf = 1.0
    q.save_fig = False
    q.custom_line_components = {}
    q.line_component_amp_median = np.array([5.0])
    q.line_component_mu_median = np.array([np.log(3000.0)])
    q.line_component_sig_median = np.array([0.08])
    q.line_component_profiles = q.line_broad[None, :]
    q.line_component_profiles_psf = np.full((0, len(lam)), np.nan)
    q.tied_line_meta = {"names": ["CIV_br"]}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=True, plot_1sigma=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    broad_lines = [line for line in ax.get_lines() if line.get_label() == "broad components"]
    assert len(broad_lines) == 1
    np.testing.assert_allclose(broad_lines[0].get_ydata(), q.line_broad, rtol=1e-12, atol=1e-12)
    plt.close(fig)


def test_plot_fig_includes_custom_line_component_trace():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.full_like(lam, 1.2)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.full_like(lam, 0.4)
    q.line_broad = np.full_like(lam, 0.4)
    q.line_narrow = np.zeros_like(lam)
    q.custom_components = {}
    q.custom_line_components = {"custom_civ_wing": np.full_like(lam, 0.4)}
    q.pred_bands = {"custom_civ_wing": (np.full_like(lam, 0.3), np.full_like(lam, 0.5))}
    q.scale_psf = 1.0
    q.save_fig = False
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.line_component_profiles = np.empty((0, len(lam)))
    q.line_component_profiles_psf = np.empty((0, len(lam)))
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=True, plot_1sigma=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "custom civ wing" in labels
    plt.close(fig)


def test_plot_fig_draws_prediction_bands_in_psf_space():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.full_like(lam, 1.0)
    q.psf_model = np.full_like(lam, 2.0)
    q.host = np.zeros_like(lam)
    q.host_psf = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.qso_psf = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.line_psf = np.zeros_like(lam)
    q.custom_components = {}
    q.custom_line_components = {}
    q.pred_bands = {"total_model": (np.full_like(lam, 0.8), np.full_like(lam, 1.2))}
    q.pred_bands_psf = {"total_model": (np.full_like(lam, 1.7), np.full_like(lam, 2.3))}
    q.scale_psf = 1.0
    q.save_fig = False
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.line_component_profiles = np.empty((0, len(lam)))
    q.line_component_profiles_psf = np.empty((0, len(lam)))
    q.tied_line_meta = {}
    q.qso_psf = np.zeros_like(lam)
    q.line_broad_psf = np.zeros_like(lam)
    q.line_narrow_psf = np.zeros_like(lam)

    q.plot_fig(show_plot=False, plot_legend=False, plot_1sigma=True, plot_psf_space=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    assert len(ax.collections) >= 1
    ymin = np.nanmin(ax.collections[0].get_paths()[0].vertices[:, 1])
    ymax = np.nanmax(ax.collections[0].get_paths()[0].vertices[:, 1])
    assert np.isclose(ymin, 1.7)
    assert np.isclose(ymax, 2.3)
    plt.close(fig)


def test_line_profile_from_components_uses_posterior_median_profiles():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.tied_line_meta = {"names": ["CIV_br", "CIV_na", "MgII_br"]}
    q.line_component_profiles = np.vstack([
        np.full_like(lam, 1.0),
        np.full_like(lam, 2.0),
        np.full_like(lam, 5.0),
    ])
    q.line_component_amp_median = np.array([100.0, 100.0, 100.0])
    q.line_component_mu_median = np.array([np.log(3000.0)] * 3)
    q.line_component_sig_median = np.array([0.1, 0.1, 0.1])

    profile = q.line_profile_from_components("CIV")

    np.testing.assert_allclose(profile, np.full_like(lam, 3.0))


def test_plot_fig_negative_custom_component_sets_negative_ylim():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.full_like(lam, 1.1)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {"bal_civ": np.full_like(lam, -0.8)}
    q.pred_bands = {
        "total_model": (np.full_like(lam, 1.0), np.full_like(lam, 1.2)),
        "host": (np.zeros_like(lam), np.zeros_like(lam)),
        "PL": (np.zeros_like(lam), np.zeros_like(lam)),
        "FeII": (np.zeros_like(lam), np.zeros_like(lam)),
        "Balmer_cont": (np.zeros_like(lam), np.zeros_like(lam)),
        "lines": (np.zeros_like(lam), np.zeros_like(lam)),
        "bal_civ": (np.full_like(lam, -0.9), np.full_like(lam, -0.7)),
    }
    q.scale_psf = 1.0
    q.save_fig = False
    q.custom_line_components = {}
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=True, plot_1sigma=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "BAL" in labels
    assert ax.get_ylim()[0] < -0.7
    plt.close(fig)


def test_plot_fig_auto_ylim_includes_strong_model_peak():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.ones_like(lam)
    q.model_total[len(lam) // 2] = 100.0
    q.host = np.zeros_like(lam)
    q.f_pl_model = q.model_total.copy()
    q.f_pl_model_intrinsic = q.model_total.copy()
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {}
    q.custom_line_components = {}
    q.pred_bands = None
    q.scale_psf = 1.0
    q.save_fig = False
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=False, plot_1sigma=False)

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_ylim()[1] > 100.0
    plt.close(fig)


def test_plot_fig_auto_ylim_ignores_single_raw_flux_outlier():
    lam, flux, err = _make_simple_spectrum()
    flux = flux.copy()
    flux[len(flux) // 2] = 1000.0
    err = err.copy()
    err[len(err) // 2] = 500.0
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.ones_like(lam)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.ones_like(lam)
    q.f_pl_model_intrinsic = np.ones_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {}
    q.custom_line_components = {}
    q.pred_bands = None
    q.scale_psf = 1.0
    q.save_fig = False
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=False, plot_1sigma=False)

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_ylim()[1] < 1000.0
    plt.close(fig)


def test_plot_fig_auto_ylim_includes_posterior_band_extrema():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.ones_like(lam)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.ones_like(lam)
    q.f_pl_model_intrinsic = np.ones_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {}
    q.custom_line_components = {}
    hi = np.full_like(lam, 1.2)
    hi[len(hi) // 2] = 50.0
    q.pred_bands = {
        "total_model": (np.full_like(lam, 0.8), hi),
    }
    q.scale_psf = 1.0
    q.save_fig = False
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=False, plot_1sigma=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_ylim()[1] > 50.0
    plt.close(fig)


def test_plot_fig_explicit_ylims_override_auto_ylim():
    lam, flux, err = _make_simple_spectrum()
    flux = flux.copy()
    flux[len(flux) // 2] = 100.0
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.ones_like(lam)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.ones_like(lam)
    q.f_pl_model_intrinsic = np.ones_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    q.custom_components = {}
    q.custom_line_components = {}
    q.pred_bands = None
    q.scale_psf = 1.0
    q.save_fig = False
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=False, plot_1sigma=False, ylims=(-2.0, 2.0))

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_ylim() == (-2.0, 2.0)
    plt.close(fig)


def test_plot_fig_draws_bal_components_individually_with_single_legend_label():
    lam, flux, err = _make_simple_spectrum()
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.1)
    q.wave = lam
    q.wave_prereduced = lam
    q.flux = flux
    q.flux_prereduced = flux
    q.err = err
    q.model_total = np.full_like(lam, 1.1)
    q.host = np.zeros_like(lam)
    q.f_pl_model = np.zeros_like(lam)
    q.f_pl_model_intrinsic = np.zeros_like(lam)
    q.f_fe_mgii_model = np.zeros_like(lam)
    q.f_fe_balmer_model = np.zeros_like(lam)
    q.f_bc_model = np.zeros_like(lam)
    q.f_line_model = np.zeros_like(lam)
    bal_civ = -0.8 * np.exp(-0.5 * ((lam - 2600.0) / 100.0) ** 2)
    bal_siiv = -0.5 * np.exp(-0.5 * ((lam - 3200.0) / 120.0) ** 2)
    q.custom_components = {"bal_civ": bal_civ, "bal_siiv": bal_siiv}
    q.pred_bands = None
    q.scale_psf = 1.0
    q.save_fig = False
    q.custom_line_components = {}
    q.line_component_amp_median = np.array([])
    q.line_component_mu_median = np.array([])
    q.line_component_sig_median = np.array([])
    q.line_component_profiles = np.empty((0, len(lam)))
    q.line_component_profiles_psf = np.empty((0, len(lam)))
    q.tied_line_meta = {}
    q.psf_model = np.full_like(lam, np.nan)
    q.qso_psf = np.full_like(lam, np.nan)
    q.host_psf = np.full_like(lam, np.nan)
    q.line_psf = np.full_like(lam, np.nan)

    q.plot_fig(show_plot=False, plot_legend=True, plot_1sigma=True)

    fig = plt.gcf()
    ax = fig.axes[0]
    bal_lines = [
        line for line in ax.get_lines()
        if line.get_color() == "red" and np.nanmin(line.get_ydata()) < -0.1
    ]
    labels = [line.get_label() for line in bal_lines]
    assert len(bal_lines) == 2
    assert labels.count("BAL") == 1
    np.testing.assert_allclose(bal_lines[0].get_ydata(), bal_civ)
    np.testing.assert_allclose(bal_lines[1].get_ydata(), bal_siiv)
    plt.close(fig)
