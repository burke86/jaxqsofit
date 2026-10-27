import warnings

import jax
import numpy as np
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density

from jaxsedfit.batched import BatchedInputs

from jaxqsofit import JAXQSOFit
from jaxqsofit.batch import _batch_model, _model_inputs, fit_batch


def _fitters(n_steps=100):
    """Synthetic quasar spectra at different redshifts and lengths, no host."""
    rng = np.random.default_rng(2)
    fitters = []
    for k, z in enumerate([0.2, 0.5, 0.9]):
        lam = 3800.0 * np.exp(np.arange(900 + 50 * k) * 1.0e-4 * np.log(10.0) * 4)
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
        cfg.continuum.fit_feii = True
        cfg.inference.method = "optax"
        cfg.inference.map_steps = n_steps
        for key in ("plot_fig", "save_fig", "save_result", "show_plot"):
            setattr(cfg.output, key, False)
        fitters.append(q)
    return fitters


def _prepared():
    fitters = _fitters()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preps = [q._prepare_fit(verbose=False) for q in fitters]
    table = (min(q.wave.min() for q in fitters), max(q.wave.max() for q in fitters))
    return fitters, preps, table


def test_batched_log_density_matches_single_spectrum_model():
    fitters, preps, table = _prepared()
    n_pixels = max(len(q.wave) for q in fitters)
    inputs = [_model_inputs(q, p, table, n_pixels) for q, p in zip(fitters, preps)]
    layout = BatchedInputs(_batch_model, inputs[:2])
    arrays, redshift = layout.stack(inputs)
    batched = layout.model()
    batched_lp = jax.jit(lambda params, a, z: log_density(batched, (a, z), {}, params)[0])

    for i, (q, prep, x) in enumerate(zip(fitters, preps, inputs)):
        def model(x=x):
            return _batch_model(**x)

        tr = trace(seed(model, i)).get_trace()
        params = {k: s["value"] for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]}
        expected = float(log_density(model, (), {}, params)[0])
        got = float(batched_lp(params, [a[i] for a in arrays], redshift[i]))
        assert np.isfinite(expected)
        assert abs(got - expected) < 1e-10 * max(1.0, abs(expected))

        # Masked padding pixels leave the likelihood unchanged.
        unpadded = _model_inputs(q, prep, table)
        lp = float(log_density(lambda: _batch_model(**unpadded), (), {}, params)[0])
        assert abs(lp - expected) < 1e-8 * max(1.0, abs(expected))


def test_shared_line_table_masks_uncovered_lines():
    fitters, preps, table = _prepared()
    for q, prep in zip(fitters, preps):
        meta = _model_inputs(q, prep, table)["tied_line_meta"]
        own = _model_inputs(q, prep, (q.wave.min(), q.wave.max()))["tied_line_meta"]
        active = [name for name, m in zip(meta["names"], meta["line_mask"]) if m > 0]
        assert active == list(own["names"])
        assert len(meta["names"]) >= len(active)


def test_fit_batch_returns_fitted_fitters():
    fitters = _fitters()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fit_batch(fitters, method="map", chunk_size=2, redshift_window=None, progress_bar=False)
    assert len(results) == len(fitters)
    names = None
    for q in fitters:
        assert np.isfinite(q.batch_diagnostics["loss"])
        assert "PL_norm" in q.optax_map_point or "PL_apparent_log_norm_std" in q.optax_map_point
        # One shared line table across the batch.
        names = names or list(q.tied_line_meta["names"])
        assert list(q.tied_line_meta["names"]) == names


def test_compiled_predictive_matches_numpyro_predictive(monkeypatch):
    import jax.numpy as jnp
    from numpyro.infer import Predictive

    import jaxqsofit.batch as batch
    from jaxqsofit.model import qso_fsps_joint_model

    captured = []
    load = batch._load_result

    def capture(q, prep, table_range, samples, map_point, pred_out, *args):
        captured.append((q, prep, table_range, samples, pred_out))
        return load(q, prep, table_range, samples, map_point, pred_out, *args)

    monkeypatch.setattr(batch, "_load_result", capture)
    fitters = _fitters()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_batch(fitters, method="laplace", chunk_size=2, num_samples=8, redshift_window=None, progress_bar=False)
    assert len(captured) == len(fitters)
    for q, prep, table_range, samples, new in captured:
        run = prep["run_kwargs"]
        inputs = _model_inputs(q, prep, table_range)
        for key in ("stage1_pixel_mask", "pixel_mask", "flux", "return_line_components", "emit_deterministics"):
            inputs.pop(key)
        old = Predictive(
            qso_fsps_joint_model,
            posterior_samples={k: jnp.asarray(v) for k, v in samples.items()},
            return_sites=q._predictive_return_sites(custom_components=run["custom_components"],
                                                    custom_line_components=run["custom_line_components"]),
        )(jax.random.PRNGKey(0), flux=None, **inputs)
        assert set(old) == set(new)
        for site, value in old.items():
            value = np.asarray(value)
            assert new[site].shape == value.shape, site
            scale = max(float(np.nanmax(np.abs(value))), 1e-30)
            assert np.nanmax(np.abs(new[site] - value)) <= 1e-6 * scale, site
