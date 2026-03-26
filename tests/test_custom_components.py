import numpy as np
import matplotlib.pyplot as plt

import jaxqsofit.core as coremod
import jaxqsofit.model as modelmod
from jaxqsofit import (
    QSOFit,
    make_custom_component,
    make_template_component,
)
from jaxqsofit.custom_components import (
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
        prior_config={},
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


def test_reconstruct_posterior_spectrum_passes_custom_components(monkeypatch):
    lam, flux, err = _make_simple_spectrum()
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)
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
    q = QSOFit(lam=lam, flux=flux, err=err, z=0.1)
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
