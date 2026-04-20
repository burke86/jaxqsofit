import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_selsing_composite_fit_wrms_below_threshold(tmp_path: Path):
    """Fit Selsing composite and require Ly-alpha-masked normalized WRMS < threshold."""
    from jaxqsofit import QSOFit, build_default_prior_config

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

    prior_config = build_default_prior_config(flux, pl_pivot=3000.0)

    q = QSOFit(lam=lam, flux=flux, err=err, z=0.0)
    q.fit(
        deredden=False,
        fit_method="optax+nuts",
        fit_lines=True,
        decompose_host=False,
        fit_pl=True,
        fit_fe=True,
        fit_bc=False,
        fit_poly=True,
        mask_lya_forest=False,
        plot_fig=False,
        save_fig=False,
        save_result=False,
        prior_config=prior_config,
        optax_steps=int(os.getenv("JAXQSOFIT_SELSING_OPTAX_STEPS", "500")),
        optax_lr=float(os.getenv("JAXQSOFIT_SELSING_OPTAX_LR", "1e-2")),
        nuts_warmup=int(os.getenv("JAXQSOFIT_SELSING_NUTS_WARMUP", "30")),
        nuts_samples=int(os.getenv("JAXQSOFIT_SELSING_NUTS_SAMPLES", "30")),
        nuts_chains=1,
        nuts_target_accept=0.9,
    )

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
