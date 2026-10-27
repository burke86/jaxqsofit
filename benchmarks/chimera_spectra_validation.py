"""Chimera spectral validation with batched MAP fits.

Fits the composite spectra of the Chimera validation sample (COSMOS host galaxy
spectrum + weighted SDSS DR7 quasar spectrum; see jaxsedfit
``benchmarks/data/chimera_validation_spectra.fits``) with
:func:`jaxqsofit.fit_batch`, spectrum only, and compares the recovered stellar
mass, AGN bolometric luminosity, SFR and emission-line FWHM and flux with the
truths.

Truths
------
Stellar mass and SFR
    COSMOS2015 LePhare medians of the host (``log_stellar_mass_truth``,
    ``log_sfr_truth`` in jaxsedfit ``chimera_validation_sample.csv``).
AGN luminosity
    Shen et al. (2011) ``logLbol`` of the quasar plus log10 of the chimera QSO
    weight. The fitted value uses the same bolometric corrections on the fitted
    power-law continuum: 9.26 L5100 (z < 0.7), 5.15 L3000 (z < 1.9), else
    3.81 L1350.
Emission lines
    FWHM and flux of the broad H-alpha, H-beta and MgII lines from Wu & Shen
    (2022), matched by the quasar's SDSS name (the prefix of the
    chimera id); ``data/chimera_line_truth.csv``. The chimera line flux is their
    flux times the QSO weight and the factor that scaled the quasar spectrum to
    its photometry (moving the quasar to the host redshift leaves line fluxes
    unchanged); the FWHM does not depend on either.

Each quantity is computed for every Laplace draw (``fit_batch(method="laplace")``);
the point estimate is the median of the draws and the interval spans their 2.3rd to
97.7th percentiles. The fitted SFR is the current SFR of the delayed-tau SFH; line
FWHM and flux are measured on the sum of the line's broad components. Narrow lines are left out:
a strong narrow peak sets the half maximum of the whole line, and the host galaxy
adds its own narrow lines, which the quasar truth does not include.

Subcommands
-----------
make-truth
    Build ``data/chimera_line_truth.csv`` from the Wu & Shen (2022) catalog
    (only needed when the sample changes).
run
    Fit the sample and write ``results.json``, ``metrics.json``, the plots and a
    Markdown summary (``output``) for the PR comment. Nothing is gated yet.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TRUTH = HERE / "data" / "chimera_line_truth.csv"
LUMINOSITY_BIN_LABELS = ["L < 42", "42 ≤ L < 43", "43 ≤ L < 44", "44 ≤ L < 45"]
OUTLIER_DEX = 0.5
REDSHIFT_WINDOW = 0.25  # fit_batch's default redshift_window
SFR_MIN_LOG_SSFR = -11.0
COMMENT_MARKER = "jaxqsofit Chimera spectral validation"
C_AA = 2.99792458e18
# Shen et al. (2011) bolometric corrections: (rest wavelength [A], correction, upper redshift).
BOLOMETRIC_CORRECTIONS = ((5100, 9.26, 0.7), (3000, 5.15, 1.9), (1350, 3.81, np.inf))
# Broad lines: jaxqsofit component prefix, Wu & Shen (2022) broad-line column, rest wavelength [A], plot label.
LINES = (
    ("Ha_br", "HALPHA_BR", 6564.61, r"H$\alpha$"),
    ("Hb_br", "HBETA_BR", 4862.68, r"H$\beta$"),
    ("MgII_br", "MGII_BR", 2798.75, "Mg II"),
)
LINE_NAMES = {"HALPHA_BR": "Hα", "HBETA_BR": "Hβ", "MGII_BR": "Mg II"}
# A line is scored when this fraction of the pixels within +-3000 km/s is unmasked.
LINE_COVERAGE = 0.8
LINE_WINDOW_LN = 0.01

# Recovered host and AGN quantities: truth column, axis label, plot file.
QUANTITIES = {
    "log_stellar_mass": dict(
        truth="log_stellar_mass_truth", title="Stellar mass",
        label=r"\log_{10}(M_\star/M_\odot)", delta=r"$\Delta\log M_\star$ [dex]", plot="chimera_spec_dmass.png",
    ),
    "log_lbol_agn": dict(
        truth="log_lbol_chimera_truth", title="AGN bolometric luminosity",
        label=r"\log_{10}(L_{\rm bol,AGN}/{\rm erg\,s^{-1}})", delta=r"$\Delta\log L_{\rm bol,AGN}$ [dex]",
        plot="chimera_spec_dlbol.png",
    ),
    "log_sfr": dict(
        truth="log_sfr_truth", title="Star formation rate",
        label=r"\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})", delta=r"$\Delta\log {\rm SFR}$ [dex]",
        plot="chimera_spec_dsfr.png",
        note="Scored on star-forming hosts (true log sSFR > −11); quiescent hosts are plotted as open circles.",
    ),
}
LINE_PLOTS = {"fwhm": "chimera_spec_line_fwhm.png", "flux": "chimera_spec_line_flux.png"}


def _sdss_name(chimera_id: str) -> str:
    """SDSS name of a chimera's quasar, e.g. ``J013549.53+241149.7``."""
    return "J" + chimera_id.split("_")[0]


def _jname_to_deg(name: str) -> tuple[float, float]:
    """RA and Dec [deg] from an SDSS name ``Jhhmmss.ss+ddmmss.s`` (truncated, not rounded)."""
    body = name[1:]
    split = max(body.find("+"), body.find("-"))
    ra, dec = body[:split], body[split:]
    ra_deg = 15.0 * (int(ra[:2]) + int(ra[2:4]) / 60.0 + float(ra[4:]) / 3600.0)
    sign = -1.0 if dec[0] == "-" else 1.0
    dec_deg = sign * (int(dec[1:3]) + int(dec[3:5]) / 60.0 + float(dec[5:]) / 3600.0)
    return ra_deg, dec_deg


# --------------------------------------------------------------------------- #
# Truth
# --------------------------------------------------------------------------- #
def make_truth(args) -> int:
    """Match the sample quasars to Wu & Shen (2022) by SDSS name (position)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.table import Table

    spectra = Table.read(args.spectra)
    ids = [str(i) for i in spectra["id"]]
    names = sorted({_sdss_name(i) for i in ids})
    targets = SkyCoord(*np.array([_jname_to_deg(n) for n in names]).T, unit=u.deg)

    columns = ["SDSS_NAME", "RA", "DEC", "Z_DR16Q", "PLATE", "MJD", "FIBERID"]
    columns += [c for _, base, _, _ in LINES for c in (base, f"{base}_ERR")]
    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        cat = SkyCoord(np.asarray(data["RA"], float), np.asarray(data["DEC"], float), unit=u.deg)
        idx, sep, _ = targets.match_to_catalog_sky(cat)
        rows = {}
        for name, i, s in zip(names, idx, sep.arcsec):
            if s > args.max_sep:
                continue
            rows[name] = {c: np.asarray(data[c][i]) for c in columns}
            rows[name]["sep_arcsec"] = float(s)

    out_rows = []
    for name in names:
        if name not in rows:
            continue
        r = rows[name]
        row = {"sdss_name": name, "wushen_name": str(r["SDSS_NAME"]).strip(), "sep_arcsec": round(r["sep_arcsec"], 3),
               "z_dr16q": float(r["Z_DR16Q"]), "plate": int(r["PLATE"]), "mjd": int(r["MJD"]), "fiberid": int(r["FIBERID"])}
        for _, base, _, _ in LINES:
            value, err = np.asarray(r[base], float), np.asarray(r[f"{base}_ERR"], float)
            # Wu & Shen line arrays: [peak wave, 50% flux centroid, flux [1e-17 erg/s/cm2], log L, FWHM, rest-frame EW]
            row[f"{base.lower()}_fwhm_kms"] = float(value[4])
            row[f"{base.lower()}_fwhm_err_kms"] = float(err[4])
            row[f"{base.lower()}_flux_1e17"] = float(value[2])
            row[f"{base.lower()}_flux_err_1e17"] = float(err[2])
        out_rows.append(row)

    out = Path(args.output)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} of {len(names)} quasars to {out} (match radius {args.max_sep}\")")
    return 0


def load_truth(sample_path, truth_path) -> dict[str, dict]:
    """Per-chimera truths keyed by chimera id (the line truths keyed by SDSS name)."""
    with Path(sample_path).open() as fh:
        sample = {r["id"]: r for r in csv.DictReader(fh)}
    lines = {}
    if truth_path is not None and Path(truth_path).exists():
        with Path(truth_path).open() as fh:
            lines = {r["sdss_name"]: r for r in csv.DictReader(fh)}
    return sample, lines


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def build_fitter(row, *, resolving_power, dsps_ssp_fn, steps):
    """Configure a spectrum-only fit of one chimera spectrum."""
    import numpyro.distributions as dist

    from jaxqsofit import JAXQSOFit
    from jaxqsofit.defaults import _build_default_prior_config

    wave = np.asarray(row["wave_obs"], dtype=float)
    to_flam = 1.0e-26 * C_AA / wave**2 / 1.0e-17  # mJy -> 1e-17 erg/s/cm2/A
    flux = np.asarray(row["flux"], dtype=float) * to_flam
    err = np.asarray(row["flux_err"], dtype=float) * to_flam
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    z = float(row["redshift"])
    q = JAXQSOFit.from_arrays(lam=wave[good], flux=flux[good], err=err[good], z=z, filename=str(row["id"]),
                              resolving_power=float(resolving_power), apply_instrumental_resolution=True)
    cfg = q.config
    cfg.observation.apply_mw_deredden = False  # both components are already dereddened
    cfg.host.dsps_ssp_fn = str(dsps_ssp_fn)
    cfg.inference.method = "optax"
    cfg.inference.map_steps = int(steps)
    # Default priors, except that the spectra are scaled to the total photometry, so the
    # host aperture scale is fixed at one.
    prior = _build_default_prior_config(flux[good] * (1.0 + z))
    prior.host.aperture_scale = dist.Normal(0.0, 0.01)
    cfg.prior_config = prior
    for key in ("plot_fig", "save_fig", "save_result", "show_plot"):
        setattr(cfg.output, key, False)
    return q


def _line(q, prefix, rest_wave):
    """FWHM [km/s] and flux [1e-17 erg/s/cm2] of a broad line per draw, or None if not covered."""
    lnw = np.log(np.asarray(q.wave, dtype=float))
    window = np.abs(lnw - np.log(rest_wave)) < LINE_WINDOW_LN
    step = float(np.median(np.diff(lnw)))
    if window.sum() < LINE_COVERAGE * 2 * LINE_WINDOW_LN / step:
        return None
    names = [str(n) for n in q.tied_line_meta["names"]]
    members = [i for i, n in enumerate(names) if n == prefix or n.startswith(f"{prefix}_")]
    if not members:
        return None
    po = q.pred_out
    amp, mu, sig = (np.asarray(po[f"line_{k}_per_component"], dtype=float).reshape(-1, len(names))[:, members]
                    for k in ("amp", "mu", "sig"))
    grid = np.log(rest_wave) + np.linspace(-0.1, 0.1, 20001)
    fwhm, flux = [], []
    for a_d, m_d, s_d in zip(amp, mu, sig):
        profile = np.sum(a_d[:, None] * np.exp(-0.5 * ((grid[None, :] - m_d[:, None]) / s_d[:, None]) ** 2), axis=0)
        # The fitted flux is in rest-frame units ((1 + z) f_lambda), so its rest-frame integral is the observed flux.
        f, area = q.line_props(profile, np.exp(grid))
        fwhm.append(f)
        flux.append(area)
    return np.asarray(fwhm, dtype=float), np.asarray(flux, dtype=float)


def _log_sfr(po):
    """Current SFR [Msun/yr] of the delayed-tau SFH per draw: SFR(t) = t exp(-t/tau) / tau^2."""
    age = np.ravel(np.asarray(po["sfh_age_gyr"], dtype=float))
    tau = np.ravel(np.asarray(po["sfh_tau_gyr"], dtype=float))
    formed = np.ravel(np.asarray(po["formed_stellar_mass"], dtype=float))
    x = age / tau
    shape = x * np.exp(-x) / tau            # SFR(age) for unit formed mass, per Gyr
    total = -np.expm1(-x) - x * np.exp(-x)  # formed mass for unit normalization
    return np.log10(formed * shape / total / 1.0e9)


def _log_lbol(po, z):
    """AGN bolometric luminosity per draw with the Shen et al. (2011) corrections."""
    for wave, correction, z_max in BOLOMETRIC_CORRECTIONS:
        if z < z_max:
            return np.ravel(np.asarray(po[f"log_lambda_Llambda_{wave}_agn"], dtype=float)) + np.log10(correction)
    return np.array([np.nan])


def _summary(values, prefix):
    """Median and 2.3/97.7 percentiles of the draws, keyed ``{prefix}_fit``/``_p02275``/``_p97725``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_fit": np.nan, f"{prefix}_p02275": np.nan, f"{prefix}_p97725": np.nan}
    lo, med, hi = np.percentile(values, [2.275, 50.0, 97.725])
    return {f"{prefix}_fit": float(med), f"{prefix}_p02275": float(lo), f"{prefix}_p97725": float(hi)}


def _release_memory() -> None:
    """Free released fitters and, on Linux, hand the memory back to the OS (glibc keeps it otherwise)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def _sorted_rows(spectra, limit=None):
    """Spectra (optionally the first ``limit``) in redshift order, so slices share line tables."""
    rows = list(spectra if not limit else spectra[:limit])
    return sorted(rows, key=lambda r: float(r["redshift"]))


def _slices(rows, max_size):
    """(start, stop) of each slice of the redshift-sorted ``rows``.

    Slices follow the redshift windows ``fit_batch`` forms (ln(1 + z) no wider than
    REDSHIFT_WINDOW), so each slice compiles one fit program and one predictive;
    windows with more than ``max_size`` spectra are split evenly.
    """
    lnz = [np.log1p(max(float(r["redshift"]), 0.0)) for r in rows]
    bounds, start = [], 0
    while start < len(rows):
        stop = start + 1
        while stop < len(rows) and lnz[stop] - lnz[start] <= REDSHIFT_WINDOW:
            stop += 1
        edges = np.linspace(start, stop, -(-(stop - start) // max_size) + 1).round().astype(int)
        bounds += [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
        start = stop
    return bounds


def fit_rows(rows, sample, lines, *, resolving_power, dsps_ssp_fn, steps, chunk, num_samples):
    """Fit ``rows`` with one ``fit_batch`` call and return the measured values and the fit time."""
    from jaxqsofit import fit_batch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitters = [build_fitter(r, resolving_power=resolving_power, dsps_ssp_fn=dsps_ssp_fn, steps=steps)
                   for r in rows]
        t0 = time.time()
        fits = fit_batch(fitters, method="laplace", chunk_size=chunk, num_samples=num_samples)
        elapsed = time.time() - t0
    return [_measure(row, q, fit, sample, lines) for row, q, fit in zip(rows, fitters, fits)], elapsed


def fit_sample(args, spectra, sample, lines):
    """Fit in redshift-window slices, keeping only the measured values of each slice.

    ``fit_batch`` keeps every fitter until it returns (about 16 MB each at 50 draws),
    so spectra are fitted one redshift window, of at most ``--slice`` spectra, at a
    time. Each window compiles its own programs anyway, so slicing along them adds
    no compiles. With ``--isolate-slices`` each slice runs in its own Python process,
    which returns all of its memory to the OS when it exits; within one process,
    compiled programs and glibc's allocator kept the memory floor rising from slice
    to slice on CI runners.
    """
    rows = _sorted_rows(spectra, args.limit)
    results, elapsed = [], 0.0
    for start, stop in _slices(rows, args.slice):
        print(f"[chimera-spectra] spectra {start + 1}-{stop} of {len(rows)}", flush=True)
        if args.isolate_slices:
            out = Path(args.output_dir) / f"slice_{start:04d}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "fit-slice", "--start", str(start), "--stop", str(stop),
                   "--out", str(out)] + _shared_cli_args(args)
            subprocess.run(cmd, check=True)
            part = json.loads(out.read_text())
            out.unlink()
            results += part["results"]
            elapsed += part["elapsed_s"]
            print(f"[chimera-spectra] fitted in {part['elapsed_s']:.0f} s", flush=True)
        else:
            part, t = fit_rows(rows[start:stop], sample, lines, resolving_power=spectra.meta["SPEC_RES"],
                               dsps_ssp_fn=args.dsps_ssp_fn, steps=args.steps, chunk=args.chunk,
                               num_samples=args.samples)
            results += part
            elapsed += t
            print(f"[chimera-spectra] fitted in {t:.0f} s", flush=True)
            _release_memory()
    return results, elapsed


def _shared_cli_args(args):
    """Command-line options a ``fit-slice`` child needs to reproduce the parent's setup."""
    out = ["--dsps-ssp-fn", args.dsps_ssp_fn, "--spectra", args.spectra, "--sample", args.sample,
           "--truth", args.truth, "--chunk", str(args.chunk), "--steps", str(args.steps), "--samples", str(args.samples)]
    return out + (["--limit", str(args.limit)] if args.limit else [])


def fit_slice(args) -> int:
    """Fit one slice (used by ``run --isolate-slices``) and write its results as JSON."""
    from astropy.table import Table

    spectra = Table.read(args.spectra)
    sample, lines = load_truth(args.sample, args.truth)
    rows = _sorted_rows(spectra, args.limit)[args.start:args.stop]
    results, elapsed = fit_rows(rows, sample, lines, resolving_power=spectra.meta["SPEC_RES"],
                                dsps_ssp_fn=args.dsps_ssp_fn, steps=args.steps, chunk=args.chunk,
                                num_samples=args.samples)
    Path(args.out).write_text(json.dumps({"results": results, "elapsed_s": elapsed}))
    return 0


def _measure(row, q, fit, sample, lines):
    """Fitted and true values for one chimera."""
    object_id = str(row["id"])
    truth = sample[object_id]
    z = float(row["redshift"])
    po = q.pred_out
    result = {
        "id": object_id,
        "sdss_name": _sdss_name(object_id),
        "redshift": z,
        "galaxy_survey": str(row["galaxy_survey"]),
        "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
        "luminosity_bin_index": int(truth["luminosity_bin_index"]),
        "log_stellar_mass_truth": float(truth["log_stellar_mass_truth"]),
        "log_stellar_mass_map": float(np.ravel(q.optax_map_point["log_stellar_mass"])[0]),
        **_summary(np.ravel(fit.samples["log_stellar_mass"]), "log_stellar_mass"),
        "log_lbol_agn_truth": float(truth["log_lbol_chimera_truth"]),
        **_summary(_log_lbol(po, z), "log_lbol_agn"),
        "log_sfr_truth": float(truth["log_sfr_truth"]),
        **_summary(_log_sfr(po), "log_sfr"),
        "map_loss": float(q.batch_diagnostics["loss"]),
        "hessian_min_eigenvalue": q.batch_diagnostics["hessian_min_eigenvalue"],
    }
    line_truth = lines.get(result["sdss_name"], {})
    # Chimera line flux = quasar line flux x QSO weight x scale of the quasar spectrum to its photometry.
    flux_scale = float(row["chimera_QSO_weight"]) * float(row["qso_scale"])
    for prefix, base, rest_wave, _ in LINES:
        measured = _line(q, prefix, rest_wave)
        key = base.lower()
        for kind, values in zip(("fwhm", "flux"), measured if measured is not None else ([np.nan], [np.nan])):
            result.update(_summary(values, f"{key}_{kind}"))
        result[f"{key}_fwhm_truth"] = float(line_truth.get(f"{key}_fwhm_kms", np.nan))
        result[f"{key}_flux_truth"] = float(line_truth.get(f"{key}_flux_1e17", np.nan)) * flux_scale
    return result


# --------------------------------------------------------------------------- #
# Metrics, plots, comment
# --------------------------------------------------------------------------- #
def _stats(res):
    res = np.asarray(res, dtype=float)
    finite = res[np.isfinite(res)]
    if finite.size == 0:
        return {"n": int(res.size), "n_failed": int(res.size), "median": np.nan, "nmad": np.nan, "rms": np.nan,
                "outlier_fraction": np.nan}
    return {
        "n": int(res.size),
        "n_failed": int(res.size - finite.size),
        "median": float(np.median(finite)),
        "nmad": float(1.4826 * np.median(np.abs(finite - np.median(finite)))),
        "rms": float(np.sqrt(np.mean(finite**2))),
        # Failed fits count as outliers.
        "outlier_fraction": float((np.sum(np.abs(finite) > OUTLIER_DEX) + res.size - finite.size) / max(res.size, 1)),
    }


def _scored(result, name) -> bool:
    # COSMOS2015 SFRs of quiescent hosts are effectively upper limits.
    if name == "log_sfr":
        return result["log_sfr_truth"] - result["log_stellar_mass_truth"] > SFR_MIN_LOG_SSFR
    return True


def _coverage(rows, prefix):
    """Fraction of rows whose truth lies inside the 95% interval of ``prefix``."""
    inside = [r[f"{prefix}_p02275"] <= r[f"{prefix}_truth"] <= r[f"{prefix}_p97725"] for r in rows
              if np.isfinite(r[f"{prefix}_p02275"])]
    return float(np.mean(inside)) if inside else np.nan


def _line_pairs(results, base, kind):
    """(truth, fit, result) for chimeras whose line is covered and has a Wu & Shen value."""
    key = base.lower()
    out = []
    for r in results:
        truth, fit = r[f"{key}_{kind}_truth"], r[f"{key}_{kind}_fit"]
        if np.isfinite(truth) and truth > 0 and np.isfinite(fit):
            out.append((truth, fit, r))
    return out


def compute_metrics(results) -> dict:
    metrics = {}
    for name in QUANTITIES:
        metrics[name] = []
        for b, label in enumerate(LUMINOSITY_BIN_LABELS):
            rows = [r for r in results if r["luminosity_bin_index"] == b and _scored(r, name)]
            res = [r[f"{name}_fit"] - r[f"{name}_truth"] for r in rows]
            metrics[name].append({"bin": b, "label": label, **_stats(res), "coverage": _coverage(rows, name)})
    for kind in LINE_PLOTS:
        metrics[f"line_{kind}"] = []
        for _, base, _, _ in LINES:
            for b, bin_label in enumerate(LUMINOSITY_BIN_LABELS):
                pairs = [(t, f, r) for t, f, r in _line_pairs(results, base, kind) if r["luminosity_bin_index"] == b]
                res = [np.log10(f / t) for t, f, _ in pairs]
                metrics[f"line_{kind}"].append({"line": base, "line_label": LINE_NAMES[base], "bin": b,
                                                "label": bin_label, **_stats(res),
                                                "coverage": _coverage([r for _, _, r in pairs], f"{base.lower()}_{kind}")})
    return metrics


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jaxqsofit.mplstyle import style_path

    plt.style.use(style_path())
    return plt


def plot_quantity(results, metrics, name, path: Path) -> None:
    plt = _style()
    from matplotlib.colors import LogNorm

    q = QUANTITIES[name]
    ylim = 3.0
    truth = np.array([r[f"{name}_truth"] for r in results])
    resid = np.array([r[f"{name}_fit"] for r in results]) - truth
    lo95 = np.array([r[f"{name}_p02275"] for r in results]) - truth
    hi95 = np.array([r[f"{name}_p97725"] for r in results]) - truth
    weight = np.array([r["chimera_QSO_weight"] for r in results])
    bins = np.array([r["luminosity_bin_index"] for r in results])
    scored = np.array([_scored(r, name) for r in results])
    norm = LogNorm(vmin=weight[weight > 0].min(), vmax=weight.max())
    clipped = np.abs(resid) > 0.97 * ylim  # drawn at the edge as triangles
    shown = np.clip(resid, -0.97 * ylim, 0.97 * ylim)

    fig, axes = plt.subplots(1, len(metrics), figsize=(17.0, 4.2), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sel = (bins == m["bin"]) & np.isfinite(resid)
        ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=0)
        for y in (OUTLIER_DEX, -OUTLIER_DEX):
            ax.axhline(y, color="0.55", lw=0.9, ls=":", zorder=0)
        bars = sel & scored & ~clipped
        # The median can fall outside a skewed interval's percentiles only by rounding; clip at zero.
        yerr = np.clip(np.vstack([resid[bars] - lo95[bars], hi95[bars] - resid[bars]]), 0.0, None)
        ax.errorbar(truth[bars], resid[bars], yerr=yerr, fmt="none", ecolor="0.55", elinewidth=0.8, capsize=1.5,
                    alpha=0.55, zorder=1)
        for subset, marker in ((sel & scored & ~clipped, "o"), (sel & scored & clipped, "^")):
            scatter = ax.scatter(truth[subset], shown[subset], c=weight[subset], cmap="viridis", norm=norm, s=42,
                                 alpha=0.9, marker=marker, edgecolor="black", linewidth=0.35, zorder=2)
        if not scored.all():
            ax.scatter(truth[sel & ~scored], shown[sel & ~scored], s=30, facecolor="none", edgecolor="0.45",
                       linewidth=0.7, zorder=1.5, label="quiescent host (not scored)")
        ax.set_title(f"{m['label']}  (N={m['n']})\noutliers={100 * m['outlier_fraction']:.1f}%, "
                     f"RMS={m['rms']:.2f} dex", fontsize=11)
        ax.set_xlabel(f"True ${q['label']}$")
        ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.18)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel(q["delta"])
    if not scored.all():
        first.legend(loc="lower left", fontsize=9)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Chimera QSO weight", pad=0.01)
    fig.savefig(path, dpi=110)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_lines(results, kind, path: Path) -> None:
    plt = _style()

    if kind == "fwhm":
        x_label, y_label, unit = "Wu & Shen (2022) FWHM", "Fitted FWHM", "km s$^{-1}$"
    else:
        x_label, y_label = "Wu & Shen (2022) flux × QSO weight", "Fitted line flux"
        unit = r"$10^{-17}$ erg s$^{-1}$ cm$^{-2}$"
    colors = ["#1f3b73", "#2a78d6", "#eda100", "#eb6834"]  # one per AGN luminosity bin, faint to bright
    fig, axes = plt.subplots(1, len(LINES), figsize=(15.0, 4.8), constrained_layout=True)
    for ax, (_, base, _, label) in zip(axes, LINES):
        pairs = _line_pairs(results, base, kind)
        truths = np.array([t for t, _, _ in pairs] or [1.0])
        lo, hi = truths.min() / 10**0.75, truths.max() * 10**0.75  # fits beyond this are drawn at the edge
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", zorder=0)
        for factor in (10**OUTLIER_DEX, 10**-OUTLIER_DEX):
            ax.plot([lo, hi], [lo * factor, hi * factor], color="0.55", lw=0.9, ls=":", zorder=0)
        for b, bin_label in enumerate(LUMINOSITY_BIN_LABELS):
            sel = [(t, f, r) for t, f, r in pairs if r["luminosity_bin_index"] == b]
            if sel:
                key = f"{base.lower()}_{kind}"
                t = np.array([x[0] for x in sel])
                f = np.array([x[1] for x in sel])
                f_lo = np.array([x[2][f"{key}_p02275"] for x in sel])
                f_hi = np.array([x[2][f"{key}_p97725"] for x in sel])
                inside = (f > lo) & (f < hi)
                yerr = np.clip(np.vstack([f - f_lo, f_hi - f]), 0.0, None)[:, inside]
                ax.errorbar(t[inside], f[inside], yerr=yerr, fmt="none", ecolor=colors[b], elinewidth=0.8,
                            alpha=0.45, zorder=1)
                ax.scatter(t, np.clip(f, lo, hi), s=36, color=colors[b], edgecolor="black", linewidth=0.35,
                           alpha=0.9, label=bin_label, zorder=2)
        res = [np.log10(f / t) for t, f, _ in pairs]
        stats = _stats(res)
        ax.set_title(f"{label} (N={stats['n']})  median Δ={stats['median']:+.2f} dex, NMAD={stats['nmad']:.2f}",
                     fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{x_label} [{unit}]")
        ax.set_ylabel(f"{y_label} [{unit}]")
        ax.grid(alpha=0.18, which="both")
    axes[0].legend(loc="upper left", fontsize=9, title="Injected log L")
    fig.savefig(path, dpi=110)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _table(rows, first_col, first_key):
    lines = [f"| {first_col} | N | median Δ | NMAD | RMS | outliers | in 95% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        if m["n"] == 0:
            continue
        lines.append(f"| {m[first_key]} | {m['n']} | {m['median']:+.3f} | {m['nmad']:.3f} | {m['rms']:.3f} | "
                     f"{100 * m['outlier_fraction']:.1f}% | {100 * m['coverage']:.0f}% |")
    return lines


def write_comment(path: Path, metrics, *, n_objects, sha, elapsed, image_base_url) -> None:
    def image(name, alt):
        return f"![{alt}]({image_base_url}/{name})" if image_base_url else "_Plot attached to the workflow artifacts._"

    lines = [
        f"## {COMMENT_MARKER}",
        "",
        f"Commit `{sha}` · {n_objects} Chimera spectra · spectrum-only MAP + Laplace (`fit_batch`) in {elapsed / 60:.1f} min · "
        "reported, not gated",
        "",
        f"Residual = posterior median − truth in bins of injected AGN luminosity (log erg/s); outlier: |Δ| > {OUTLIER_DEX} dex; "
        "“in 95%”: fraction of truths inside the Laplace 95% interval (statistical errors only).",
    ]
    for name, q in QUANTITIES.items():
        lines += ["", f"### {q['title']}", ""]
        if q.get("note"):
            lines += [q["note"], ""]
        lines += _table(metrics[name], "AGN luminosity bin", "label") + ["", image(q["plot"], q["title"])]
    for kind, title in (("fwhm", "Broad-line FWHM"), ("flux", "Broad-line flux")):
        lines += ["", f"### {title} vs Wu & Shen (2022)", "",
                  "Δ = log10(fit / truth), for chimeras whose line is covered and measured by Wu & Shen; "
                  "the true flux is theirs × QSO weight × the quasar's photometric scale. jaxqsofit's broad Gaussians "
                  "have FWHM ≥ 2,800 km/s, so narrower Wu & Shen broad lines cannot be matched.", ""]
        rows = [{**m, "name": f"{m['line_label']}, {m['label']}"} for m in metrics[f"line_{kind}"]]
        lines += _table(rows, "Line, AGN luminosity bin", "name") + ["", image(LINE_PLOTS[kind], title)]
    workflow_url = os.environ.get("GITHUB_SERVER_URL", "") and (
        f"{os.environ['GITHUB_SERVER_URL']}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    if workflow_url:
        lines += ["", f"[Workflow run]({workflow_url})"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> int:
    from astropy.table import Table

    spectra = Table.read(args.spectra)
    sample, lines = load_truth(args.sample, args.truth)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[chimera-spectra] fitting {len(_sorted_rows(spectra, args.limit))} spectra", flush=True)

    results, elapsed = fit_sample(args, spectra, sample, lines)
    metrics = compute_metrics(results)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "metrics.json").write_text(json.dumps({**metrics, "elapsed_s": elapsed}, indent=2))
    for name, q in QUANTITIES.items():
        plot_quantity(results, metrics[name], name, out / q["plot"])
    for kind, name in LINE_PLOTS.items():
        plot_lines(results, kind, out / name)
    write_comment(out / "output", metrics, n_objects=len(results), sha=args.sha, elapsed=elapsed,
                  image_base_url=args.image_base_url)
    print((out / "output").read_text())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-truth", help="Build the emission-line truth table from Wu & Shen (2022).")
    p.add_argument("--catalog", required=True, help="Wu & Shen (2022) dr16q_prop FITS catalog.")
    p.add_argument("--spectra", required=True, help="jaxsedfit chimera_validation_spectra.fits.")
    p.add_argument("--output", default=str(DEFAULT_TRUTH))
    p.add_argument("--max-sep", type=float, default=1.0, help="Match radius [arcsec].")
    p.set_defaults(func=make_truth)

    def add_fit_args(p):
        p.add_argument("--dsps-ssp-fn", required=True)
        p.add_argument("--spectra", required=True, help="jaxsedfit chimera_validation_spectra.fits.")
        p.add_argument("--sample", required=True, help="jaxsedfit chimera_validation_sample.csv (truths).")
        p.add_argument("--truth", default=str(DEFAULT_TRUTH), help="Emission-line truth table.")
        p.add_argument("--chunk", type=int, default=16, help="Spectra per compiled batch.")
        p.add_argument("--steps", type=int, default=2000, help="MAP steps.")
        p.add_argument("--samples", type=int, default=100, help="Laplace draws per spectrum.")
        p.add_argument("--limit", type=int, default=None, help="Fit only the first N spectra of the file (quick look).")

    p = sub.add_parser("run", help="Fit the spectra and write metrics, plots and the comment.")
    p.add_argument("--output-dir", required=True)
    add_fit_args(p)
    p.add_argument("--slice", type=int, default=128,
                   help="Most spectra per fit_batch call; slices follow fit_batch's redshift windows (see --isolate-slices).")
    p.add_argument("--isolate-slices", action="store_true",
                   help="Fit each slice in its own process, so its memory is returned to the OS (for CI runners).")
    p.add_argument("--sha", default="local")
    p.add_argument("--image-base-url", default=None, help="URL of the directory holding the plots.")
    p.set_defaults(func=run)

    p = sub.add_parser("fit-slice", help="Internal: fit one slice of a run --isolate-slices and write JSON.")
    add_fit_args(p)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=fit_slice)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
