import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pytest

from jaxqsofit import JAXQSOFit
from jaxqsofit.config import PriorConfig

pytestmark = pytest.mark.integration


def test_selsing_composite_fit_wrms_below_threshold(tmp_path: Path):
    """Fit Selsing composite and require Ly-alpha-masked normalized WRMS < threshold."""
    url = "https://raw.githubusercontent.com/jselsing/QuasarComposite/master/Selsing2015.dat"
    dat_path = tmp_path / "Selsing2015.dat"

    try:
        urlretrieve(url, dat_path)
    except Exception as exc:
        pytest.skip(f"Selsing composite download unavailable: {exc}")

    arr = np.loadtxt(dat_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        pytest.skip("Unexpected Selsing composite format")

    lam = np.asarray(arr[:, 0], dtype=float)
    flux = np.asarray(arr[:, 1], dtype=float)

    if arr.shape[1] >= 3:
        err = np.asarray(arr[:, 2], dtype=float)
    else:
        err = np.full_like(flux, 1e-3 * max(np.nanmedian(np.abs(flux)), 1e-6), dtype=float)

    m = np.isfinite(lam) & np.isfinite(flux) & np.isfinite(err) & (lam > 0) & (err > 0)
    lam, flux, err = lam[m], flux[m], err[m]

    if lam.size < 200:
        pytest.skip("Not enough valid composite pixels")

    prior_config = PriorConfig.from_spectrum(flux=flux, pl_pivot=3000.0)

    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=0.0)
    q.config.inference.method = "optax"
    q.config.inference.random_seed = 0
    q.config.observation.apply_mw_deredden = False
    q.config.preprocessing.mask_lya_forest = True
    q.config.lines.enabled = True
    q.config.host.enabled = False
    q.config.continuum.fit_power_law = True
    q.config.continuum.fit_feii = True
    q.config.continuum.fit_balmer_continuum = False
    q.config.continuum.fit_polynomial_tilt = True
    q.config.output.plot_fig = False
    q.config.output.save_fig = False
    q.config.output.save_result = False
    q.config.prior_config = prior_config
    q.config.inference.map_steps = int(os.getenv("JAXQSOFIT_SELSING_OPTAX_STEPS", "1200"))
    q.config.inference.learning_rate = float(os.getenv("JAXQSOFIT_SELSING_OPTAX_LR", "1e-2"))
    q.config.inference.num_warmup = int(os.getenv("JAXQSOFIT_SELSING_NUTS_WARMUP", "30"))
    q.config.inference.num_samples = int(os.getenv("JAXQSOFIT_SELSING_NUTS_SAMPLES", "30"))
    q.config.inference.num_chains = 1
    q.config.inference.target_accept_prob = 0.9
    q.fit()

    resid = np.asarray(q.flux, dtype=float) - np.asarray(q.model_total, dtype=float)
    sigma = np.asarray(q.err, dtype=float)

    # Include inferred jitter terms in effective sigma when available.
    if getattr(q, "numpyro_samples", None) is not None:
        s = q.numpyro_samples
        frac_j = float(np.median(np.asarray(s["frac_jitter"]))) if "frac_jitter" in s else 0.0
        add_j = float(np.median(np.asarray(s["add_jitter"]))) if "add_jitter" in s else 0.0
        sigma = np.sqrt(sigma**2 + (frac_j * np.abs(np.asarray(q.model_total)))**2 + add_j**2)

    # Ly-alpha masked metric: exclude wavelengths bluer than 1215.67 A.
    mfit = (
        np.isfinite(resid)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(np.asarray(q.wave, dtype=float))
        & (np.asarray(q.wave, dtype=float) >= 1215.67)
    )

    if np.sum(mfit) < 50:
        pytest.skip("Not enough Ly-alpha-masked pixels to evaluate WRMS")

    zres = resid[mfit] / sigma[mfit]
    wrms = float(np.sqrt(np.mean(zres**2)))

    threshold = float(os.getenv("JAXQSOFIT_SELSING_WRMS_THRESHOLD", "2.0"))
    assert wrms < threshold, f"Selsing WRMS too high: {wrms:.3f} (threshold={threshold:.3f})"
