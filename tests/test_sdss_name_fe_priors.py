import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_sdss_name_fe_width_prior_wrms_below_threshold():
    """Resolve a named target, fit with tight Fe-FWHM priors, and require WRMS < 1.8."""
    astroquery = pytest.importorskip("astroquery.sdss")
    coordinates = pytest.importorskip("astropy.coordinates")
    units = pytest.importorskip("astropy.units")

    SDSS = astroquery.SDSS
    SkyCoord = coordinates.SkyCoord
    u = units

    try:
        coord = SkyCoord.from_name("J102839.11+450009.4")
    except Exception as exc:
        pytest.skip(f"Name resolution unavailable: {exc}")

    try:
        xid = SDSS.query_region(coord, radius=5 * u.arcsec, spectro=True)
    except Exception as exc:
        pytest.skip(f"SDSS query unavailable: {exc}")

    if xid is None or len(xid) == 0:
        pytest.skip("No SDSS spectrum found for named target")

    try:
        spectra = SDSS.get_spectra(matches=xid[:1])
    except Exception as exc:
        pytest.skip(f"SDSS spectrum download unavailable: {exc}")

    if not spectra:
        pytest.skip("No SDSS spectra returned")

    from jaxqsofit import QSOFit, build_default_prior_config

    hdu = spectra[0]
    data = hdu[1].data
    lam = np.asarray(10 ** data["loglam"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    ivar = np.asarray(data["ivar"], dtype=float)

    err = np.full_like(flux, 1e-6, dtype=float)
    good = np.isfinite(ivar) & (ivar > 0)
    err[good] = 1.0 / np.sqrt(ivar[good])
    err[~np.isfinite(err)] = 1e-6
    err[err <= 0] = 1e-6

    z = float(xid[0]["z"]) if "z" in xid.colnames else 0.1

    prior_config = build_default_prior_config(flux)
    prior_config["log_Fe_uv_FWHM"] = {"loc": np.log(100.0), "scale": 0.1}
    prior_config["log_Fe_op_FWHM"] = {"loc": np.log(100.0), "scale": 0.1}

    q = QSOFit(lam=lam, flux=flux, err=err, z=z, ra=float(coord.ra.deg), dec=float(coord.dec.deg))
    q.fit(
        deredden=False,
        fit_method="optax+nuts",
        fit_lines=True,
        decompose_host=False,
        fit_fe=True,
        fit_bc=False,
        fit_poly=True,
        plot_fig=False,
        save_fig=False,
        save_result=False,
        prior_config=prior_config,
        optax_steps=int(os.getenv("JAXQSOFIT_NAME_OPTAX_STEPS", "300")),
        optax_lr=float(os.getenv("JAXQSOFIT_NAME_OPTAX_LR", "1e-2")),
        nuts_warmup=int(os.getenv("JAXQSOFIT_NAME_NUTS_WARMUP", "25")),
        nuts_samples=int(os.getenv("JAXQSOFIT_NAME_NUTS_SAMPLES", "25")),
        nuts_chains=1,
        nuts_target_accept=0.9,
    )

    resid = np.asarray(q.flux, dtype=float) - np.asarray(q.model_total, dtype=float)
    sigma = np.asarray(q.err, dtype=float)

    if getattr(q, "numpyro_samples", None) is not None:
        s = q.numpyro_samples
        frac_j = float(np.median(np.asarray(s["frac_jitter"]))) if "frac_jitter" in s else 0.0
        add_j = float(np.median(np.asarray(s["add_jitter"]))) if "add_jitter" in s else 0.0
        sigma = np.sqrt(sigma**2 + (frac_j * np.abs(np.asarray(q.model_total)))**2 + add_j**2)

    m = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if np.sum(m) < 10:
        pytest.skip("Not enough valid pixels to evaluate WRMS")

    zres = resid[m] / sigma[m]
    wrms = float(np.sqrt(np.mean(zres**2)))
    threshold = float(os.getenv("JAXQSOFIT_NAME_WRMS_THRESHOLD", "1.8"))
    assert wrms < threshold, f"WRMS too high: {wrms:.3f} (threshold={threshold:.3f})"
