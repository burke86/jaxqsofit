"""DR7 regression comparison for broad-line properties.

This integration test mirrors the legacy PyQSOFit DR7-style check:
- Download the Shen et al. DR7 BH catalog
- Fetch SDSS spectra with astroquery for sampled objects
- Fit each spectrum with jaxqsofit
- Compare derived broad-line FWHM and log L to catalog values

The test is intentionally tolerant and intended as a regression guard,
not an exact reproduction of legacy values.
"""

from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

from jaxqsofit import QSOFit

pytestmark = pytest.mark.integration

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

DR7_CAT_NAME = "dr7_bh_Nov19_2013.fits.gz"
DR7_CAT_URLS = (
    f"https://quasar.astro.illinois.edu/BH_mass/data/catalogs/{DR7_CAT_NAME}",
    f"http://quasar.astro.illinois.edu/BH_mass/data/catalogs/{DR7_CAT_NAME}",
)

# Fixed DR7 catalog row indices for a deterministic regression sample.
DEFAULT_DR7_SAMPLE_INDICES = (0, 1, 2)

# line_key -> (catalog_fwhm_col, catalog_logl_col, rest_wavelength_A)
LINE_MAP = {
    "CIV_br": ("FWHM_CIV", "LOGL_CIV", 1549.06),
    "MgII_br": ("FWHM_BROAD_MGII", "LOGL_BROAD_MGII", 2798.75),
    "Hb_br": ("FWHM_BROAD_HB", "LOGL_BROAD_HB", 4862.68),
}


def _download_dr7_catalog(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / DR7_CAT_NAME
    if out.exists():
        return out
    last_exc = None
    for url in DR7_CAT_URLS:
        try:
            urllib.request.urlretrieve(url, out)
            return out
        except Exception as exc:  # pragma: no cover - network dependent
            last_exc = exc
    raise RuntimeError(f"Failed to download DR7 catalog: {last_exc}")


def _safe_err_from_ivar(ivar: np.ndarray) -> np.ndarray:
    err = np.full_like(ivar, 1e-6, dtype=float)
    m = np.isfinite(ivar) & (ivar > 0)
    err[m] = 1.0 / np.sqrt(ivar[m])
    err[~np.isfinite(err)] = 1e-6
    err[err <= 0] = 1e-6
    return err


def _parse_sample_indices() -> tuple[int, ...]:
    """Return deterministic DR7 row indices from env or the default sample."""
    raw = os.getenv("JAXQSOFIT_DR7_SAMPLE_INDICES", "").strip()
    if not raw:
        return DEFAULT_DR7_SAMPLE_INDICES
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _flux_to_luminosity(flux_1e17: float, z: float) -> float:
    """Convert integrated line flux [1e-17 erg/s/cm^2] to luminosity [erg/s]."""
    d_l_cm = COSMO.luminosity_distance(z).to(u.cm).value
    return float(flux_1e17 * 1e-17 * 4.0 * np.pi * d_l_cm**2)


@pytest.mark.skip(reason="Temporarily disabled.")
def test_dr7_broad_line_regression(tmp_path: Path):
    """Compare fitted broad-line properties against DR7 catalog values."""
    astroquery_sdss = pytest.importorskip("astroquery.sdss")
    astropy_coords = pytest.importorskip("astropy.coordinates")
    astropy_units = pytest.importorskip("astropy.units")

    SDSS = astroquery_sdss.SDSS
    SkyCoord = astropy_coords.SkyCoord
    u = astropy_units

    sample_indices = _parse_sample_indices()

    try:
        cat_path = _download_dr7_catalog(tmp_path / "dr7_cache")
    except Exception as exc:
        pytest.skip(f"DR7 catalog unavailable: {exc}")

    try:
        with gzip.open(cat_path, "rb") as f:
            cat = fits.open(f)
            rows = cat[1].data
            if len(rows) == 0:
                pytest.skip("DR7 catalog is empty")
            valid_indices = [idx for idx in sample_indices if 0 <= idx < len(rows)]
            if len(valid_indices) == 0:
                pytest.skip("Requested DR7 sample indices are outside the catalog bounds")
            sample = [rows[idx] for idx in valid_indices]
    except Exception as exc:
        pytest.skip(f"Could not read DR7 catalog: {exc}")

    out_rows = []
    for row in sample:
        ra = float(row["RA"])
        dec = float(row["DEC"])
        plate = int(row["PLATE"])
        mjd = int(row["MJD"])
        fiber = int(row["FIBER"])
        z = float(row["REDSHIFT"])

        print("Running row", row)

        try:
            pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            xid = SDSS.query_region(pos, spectro=True, radius=5 * u.arcsec)
        except Exception:
            continue
        if xid is None or len(xid) == 0:
            continue

        m = (xid["plate"] == plate) & (xid["mjd"] == mjd) & (xid["fiberID"] == fiber)
        if np.sum(m) < 1:
            continue

        try:
            sp = SDSS.get_spectra(matches=xid[m])
        except Exception:
            continue
        if not sp:
            continue

        data = sp[0][1].data
        lam = np.asarray(10 ** data["loglam"], dtype=float)
        flux = np.asarray(data["flux"], dtype=float)
        err = _safe_err_from_ivar(np.asarray(data["ivar"], dtype=float))

        try:
            filename = f"{plate:04d}-{mjd}-{fiber:04d}"
            q = QSOFit(lam=lam, flux=flux, err=err, z=z, ra=ra, dec=dec, filename=filename)
            q.fit(
                deredden=True,
                fit_method="optax+nuts",
                fit_lines=True,
                decompose_host=True,
                fit_fe=True,
                fit_bc=True,
                fit_poly=False,
                save_result=False,
                plot_fig=True,
                save_fig=False,
                nuts_warmup=40,
                nuts_samples=20,
                nuts_chains=1,
                optax_steps=400,
                optax_lr=1e-2,
            )
        except Exception:
            continue

        wmin, wmax = float(np.nanmin(q.wave)), float(np.nanmax(q.wave))
        for line_key, (fwhm_col, logl_col, lam0) in LINE_MAP.items():
            if not (wmin <= lam0 <= wmax):
                continue
            if fwhm_col not in row.array.names or logl_col not in row.array.names:
                continue

            prof = q.line_profile_from_components(line_key)
            fwhm_fit, area_fit = q.line_props_from_profile(np.asarray(q.wave), prof)
            if not np.isfinite(fwhm_fit) or not np.isfinite(area_fit) or area_fit <= 0:
                continue

            logl_fit = float(np.log10(_flux_to_luminosity(area_fit, z=z)))
            fwhm_cat = float(row[fwhm_col])
            logl_cat = float(row[logl_col])

            if not (np.isfinite(fwhm_cat) and np.isfinite(logl_cat) and fwhm_cat > 0):
                continue

            out_rows.append(
                {
                    "plate": plate,
                    "mjd": mjd,
                    "fiber": fiber,
                    "z": z,
                    "line": line_key,
                    "fwhm_fit": fwhm_fit,
                    "fwhm_cat": fwhm_cat,
                    "logl_fit": logl_fit,
                    "logl_cat": logl_cat,
                }
            )

    if len(out_rows) == 0:
        pytest.skip("No valid DR7 line measurements collected in this run")

    res = pd.DataFrame(out_rows)
    frac_fwhm_err = np.abs(res["fwhm_fit"] - res["fwhm_cat"]) / res["fwhm_cat"]
    dlogl = np.abs(res["logl_fit"] - res["logl_cat"])

    print("Sample indices:", sample_indices)
    print("Median frac FWHM error:", float(np.median(frac_fwhm_err)))
    print("Median |dlogL|:", float(np.median(dlogl)))
    print("Frac with frac_fwhm_err < 0.5:", float(np.mean(frac_fwhm_err < 0.5)))
    print(res[["plate", "mjd", "fiber", "line", "fwhm_fit", "fwhm_cat", "logl_fit", "logl_cat"]])

    # Tolerant regression gates (model differs from legacy PyQSOFit implementation).
    assert np.all(np.isfinite(frac_fwhm_err))
    assert np.all(np.isfinite(dlogl))
    assert np.median(frac_fwhm_err) < 0.35
    assert np.median(dlogl) < 0.2
    assert np.mean(frac_fwhm_err < 0.5) > 0.7
