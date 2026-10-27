import warnings

import numpy as np

from jaxqsofit import JAXQSOFit


def _fitter(method, num_samples=40):
    """Synthetic quasar spectrum with Mg II and H-beta, no host."""
    rng = np.random.default_rng(3)
    z = 0.5
    lam = 3800.0 * np.exp(np.arange(900) * 1.0e-4 * np.log(10.0) * 4)
    wave = lam / (1.0 + z)
    flux = 20.0 * (wave / 4000.0) ** -1.5
    for center in (2798.0, 4862.68):
        flux = flux + 15.0 * np.exp(-0.5 * ((wave - center) / (center * 0.006)) ** 2)
    err = np.full_like(flux, 0.5)
    flux = flux + err * rng.normal(size=flux.size)
    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=z, ra=0.0, dec=0.0)
    cfg = q.config
    cfg.observation.apply_mw_deredden = False
    cfg.host.enabled = False
    cfg.inference.method = method
    cfg.inference.map_steps = 300
    cfg.inference.num_samples = num_samples
    for key in ("plot_fig", "save_fig", "save_result", "show_plot"):
        setattr(cfg.output, key, False)
    return q


def _fit(method):
    q = _fitter(method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return q, q.fit(verbose=False)


def test_single_spectrum_laplace_draws_around_the_map():
    q_map, map_result = _fit("optax")
    assert np.asarray(map_result.samples["PL_slope"]).shape[0] == 1
    assert q_map.approx_result is None

    q, result = _fit("laplace")
    assert result.method == "laplace"
    slope = np.asarray(result.samples["PL_slope"])
    assert slope.shape[0] == 40
    assert np.all(np.isfinite(slope)) and np.std(slope) > 0
    # Draws are centred on the same MAP, not spread over the prior.
    slope_map = float(np.asarray(map_result.samples["PL_slope"])[0])
    assert abs(np.median(slope) - slope_map) < 3 * np.std(slope)
    assert np.std(slope) < 0.5
    assert np.isfinite(q.approx_result["hessian_min_eigenvalue"])
    assert result.spectrum.line_groups  # predictive products are built from the draws
