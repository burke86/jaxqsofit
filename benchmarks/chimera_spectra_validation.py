"""Chimera spectral validation with batched MAP fits.

Fits the composite spectra of the Chimera validation sample (COSMOS host galaxy
spectrum + weighted SDSS DR7 quasar spectrum; see jaxsedfit
``benchmarks/data/chimera_validation_spectra.fits``) with
:func:`jaxqsofit.fit_batch`, spectrum only, and compares the recovered stellar
mass, AGN bolometric luminosity, SFR and broad-line FWHM and EW with the truths.

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
Broad lines
    FWHM and rest-frame EW of broad H-alpha, H-beta and MgII from Wu & Shen
    (2022), matched by the quasar's SDSS name (the prefix of the chimera id);
    ``data/chimera_broad_line_truth.csv``. Both are independent of the QSO weight.

Point estimates are MAPs; the fitted SFR is the current SFR of the delayed-tau
SFH, and broad-line FWHM and EW are measured on the summed broad components,
with the EW relative to the power-law continuum at the line (as in PyQSOFit).

Subcommands
-----------
make-truth
    Build ``data/chimera_broad_line_truth.csv`` from the Wu & Shen (2022) catalog
    (only needed when the sample changes).
run
    Fit the sample and write ``results.json``, ``metrics.json``, the plots and a
    Markdown summary (``output``) for the PR comment. Nothing is gated yet.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TRUTH = HERE / "data" / "chimera_broad_line_truth.csv"
LUMINOSITY_BIN_LABELS = ["L < 42", "42 ≤ L < 43", "43 ≤ L < 44", "44 ≤ L < 45"]
OUTLIER_DEX = 0.5
SFR_MIN_LOG_SSFR = -11.0
COMMENT_MARKER = "jaxqsofit Chimera spectral validation"
C_AA = 2.99792458e18
# Shen et al. (2011) bolometric corrections: (rest wavelength [A], correction, upper redshift).
BOLOMETRIC_CORRECTIONS = ((5100, 9.26, 0.7), (3000, 5.15, 1.9), (1350, 3.81, np.inf))
# Broad lines: jaxqsofit parent name, Wu & Shen (2022) column, rest wavelength [A], plot label.
BROAD_LINES = (
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
LINE_PLOTS = {"fwhm": "chimera_spec_broad_fwhm.png", "ew": "chimera_spec_broad_ew.png"}


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
    columns += [c for _, base, _, _ in BROAD_LINES for c in (base, f"{base}_ERR")]
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
        for _, base, _, _ in BROAD_LINES:
            value, err = np.asarray(r[base], float), np.asarray(r[f"{base}_ERR"], float)
            # Wu & Shen line arrays: [peak wave, 50% flux centroid, flux, log L, FWHM, rest-frame EW]
            row[f"{base.lower()}_fwhm_kms"] = float(value[4])
            row[f"{base.lower()}_fwhm_err_kms"] = float(err[4])
            row[f"{base.lower()}_ew_aa"] = float(value[5])
            row[f"{base.lower()}_ew_err_aa"] = float(err[5])
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


def _broad_line(q, parent, rest_wave):
    """MAP FWHM [km/s] and EW [A] of one broad line, or NaNs if it is not covered."""
    lnw = np.log(np.asarray(q.wave, dtype=float))
    window = np.abs(lnw - np.log(rest_wave)) < LINE_WINDOW_LN
    step = float(np.median(np.diff(lnw)))
    if window.sum() < LINE_COVERAGE * 2 * LINE_WINDOW_LN / step:
        return np.nan, np.nan
    meta = q.tied_line_meta
    names = [str(n) for n in meta["names"]]
    members = [i for i, n in enumerate(names) if n.rsplit("_", 1)[0] == parent]
    if not members:
        return np.nan, np.nan
    po = q.pred_out
    amp = np.asarray(po["line_amp_per_component"], dtype=float).reshape(-1, len(names))[0, members]
    mu = np.asarray(po["line_mu_per_component"], dtype=float).reshape(-1, len(names))[0, members]
    sig = np.asarray(po["line_sig_per_component"], dtype=float).reshape(-1, len(names))[0, members]
    grid = np.log(rest_wave) + np.linspace(-0.1, 0.1, 20001)
    profile = np.sum(amp[:, None] * np.exp(-0.5 * ((grid[None, :] - mu[:, None]) / sig[:, None]) ** 2), axis=0)
    fwhm, area = q.line_props(profile, np.exp(grid))
    pl = np.interp(rest_wave, np.asarray(q.wave, dtype=float), np.asarray(po["f_pl_model"], dtype=float).reshape(-1, len(q.wave))[0])
    ew = area / pl if pl > 0 else np.nan
    return float(fwhm), float(ew)


def _log_sfr(po):
    """Current SFR [Msun/yr] of the delayed-tau SFH at the MAP: SFR(t) = t exp(-t/tau) / tau^2."""
    age = float(np.ravel(po["sfh_age_gyr"])[0])
    tau = float(np.ravel(po["sfh_tau_gyr"])[0])
    formed = float(np.ravel(po["formed_stellar_mass"])[0])
    x = age / tau
    shape = x * np.exp(-x) / tau            # SFR(age) for unit formed mass, per Gyr
    total = -np.expm1(-x) - x * np.exp(-x)  # formed mass for unit normalization
    return float(np.log10(formed * shape / total / 1.0e9))


def _log_lbol(po, z):
    for wave, correction, z_max in BOLOMETRIC_CORRECTIONS:
        if z < z_max:
            return float(np.ravel(po[f"log_lambda_Llambda_{wave}_agn"])[0] + np.log10(correction))
    return np.nan


def fit_sample(spectra, sample, lines, *, dsps_ssp_fn, steps, chunk):
    from jaxqsofit import fit_batch

    rows = list(spectra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitters = [build_fitter(r, resolving_power=spectra.meta["SPEC_RES"], dsps_ssp_fn=dsps_ssp_fn, steps=steps)
                   for r in rows]
        t0 = time.time()
        fits = fit_batch(fitters, method="map", chunk_size=chunk)
        elapsed = time.time() - t0

    results = []
    for row, q, fit in zip(rows, fitters, fits):
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
            "log_stellar_mass_fit": float(np.ravel(fit.samples["log_stellar_mass"])[0]),
            "log_lbol_agn_truth": float(truth["log_lbol_chimera_truth"]),
            "log_lbol_agn_fit": _log_lbol(po, z),
            "log_sfr_truth": float(truth["log_sfr_truth"]),
            "log_sfr_fit": _log_sfr(po),
            "map_loss": float(q.batch_diagnostics["loss"]),
        }
        line_truth = lines.get(result["sdss_name"], {})
        for parent, base, rest_wave, _ in BROAD_LINES:
            fwhm, ew = _broad_line(q, parent, rest_wave)
            key = base.lower()
            result[f"{key}_fwhm_fit"], result[f"{key}_ew_fit"] = fwhm, ew
            result[f"{key}_fwhm_truth"] = float(line_truth.get(f"{key}_fwhm_kms", np.nan))
            result[f"{key}_ew_truth"] = float(line_truth.get(f"{key}_ew_aa", np.nan))
        results.append(result)
    return results, elapsed


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
            res = [r[f"{name}_fit"] - r[f"{name}_truth"] for r in results
                   if r["luminosity_bin_index"] == b and _scored(r, name)]
            metrics[name].append({"bin": b, "label": label, **_stats(res)})
    for kind in ("fwhm", "ew"):
        metrics[f"broad_{kind}"] = []
        for _, base, _, _ in BROAD_LINES:
            for b, bin_label in enumerate(LUMINOSITY_BIN_LABELS):
                pairs = [(t, f) for t, f, r in _line_pairs(results, base, kind) if r["luminosity_bin_index"] == b]
                res = [np.log10(f / t) for t, f in pairs]
                metrics[f"broad_{kind}"].append({"line": base, "line_label": LINE_NAMES[base], "bin": b, "label": bin_label,
                                                 **_stats(res)})
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

    unit = "km s$^{-1}$" if kind == "fwhm" else r"$\AA$"
    what = "FWHM" if kind == "fwhm" else "rest-frame EW"
    colors = ["#1f3b73", "#2a78d6", "#eda100", "#eb6834"]  # one per AGN luminosity bin, faint to bright
    fig, axes = plt.subplots(1, len(BROAD_LINES), figsize=(15.0, 4.8), constrained_layout=True)
    for ax, (_, base, _, label) in zip(axes, BROAD_LINES):
        pairs = _line_pairs(results, base, kind)
        lo, hi = (1.0e3, 3.0e4) if kind == "fwhm" else (3.0, 1.0e3)
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", zorder=0)
        for factor in (10**OUTLIER_DEX, 10**-OUTLIER_DEX):
            ax.plot([lo, hi], [lo * factor, hi * factor], color="0.55", lw=0.9, ls=":", zorder=0)
        for b, bin_label in enumerate(LUMINOSITY_BIN_LABELS):
            sel = [(t, f) for t, f, r in pairs if r["luminosity_bin_index"] == b]
            if sel:
                t, f = np.array(sel).T
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
        ax.set_xlabel(f"Wu & Shen (2022) {what} [{unit}]")
        ax.set_ylabel(f"Fitted {what} [{unit}]")
        ax.grid(alpha=0.18, which="both")
    axes[0].legend(loc="upper left", fontsize=9, title="Injected log L")
    fig.savefig(path, dpi=110)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _table(rows, first_col, first_key):
    lines = [f"| {first_col} | N | median Δ | NMAD | RMS | outliers |", "|---|---:|---:|---:|---:|---:|"]
    for m in rows:
        if m["n"] == 0:
            continue
        lines.append(f"| {m[first_key]} | {m['n']} | {m['median']:+.3f} | {m['nmad']:.3f} | {m['rms']:.3f} | "
                     f"{100 * m['outlier_fraction']:.1f}% |")
    return lines


def write_comment(path: Path, metrics, *, n_objects, sha, elapsed, image_base_url) -> None:
    def image(name, alt):
        return f"![{alt}]({image_base_url}/{name})" if image_base_url else "_Plot attached to the workflow artifacts._"

    lines = [
        f"## {COMMENT_MARKER}",
        "",
        f"Commit `{sha}` · {n_objects} Chimera spectra · spectrum-only MAP (`fit_batch`) in {elapsed / 60:.1f} min · "
        "reported, not gated",
        "",
        f"Residual = MAP − truth in bins of injected AGN luminosity (log erg/s); outlier: |Δ| > {OUTLIER_DEX} dex.",
    ]
    for name, q in QUANTITIES.items():
        lines += ["", f"### {q['title']}", ""]
        if q.get("note"):
            lines += [q["note"], ""]
        lines += _table(metrics[name], "AGN luminosity bin", "label") + ["", image(q["plot"], q["title"])]
    for kind, title in (("fwhm", "Broad-line FWHM"), ("ew", "Broad-line rest-frame EW")):
        lines += ["", f"### {title} (vs Wu & Shen 2022)", "",
                  "Δ = log10(fit / truth), for chimeras whose line is covered and measured by Wu & Shen.", ""]
        rows = [{**m, "name": f"{m['line_label']}, {m['label']}"} for m in metrics[f"broad_{kind}"]]
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
    if args.limit:
        spectra = spectra[: args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[chimera-spectra] fitting {len(spectra)} spectra", flush=True)

    results, elapsed = fit_sample(spectra, sample, lines, dsps_ssp_fn=args.dsps_ssp_fn, steps=args.steps,
                                  chunk=args.chunk)
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

    p = sub.add_parser("make-truth", help="Build the broad-line truth table from Wu & Shen (2022).")
    p.add_argument("--catalog", required=True, help="Wu & Shen (2022) dr16q_prop FITS catalog.")
    p.add_argument("--spectra", required=True, help="jaxsedfit chimera_validation_spectra.fits.")
    p.add_argument("--output", default=str(DEFAULT_TRUTH))
    p.add_argument("--max-sep", type=float, default=1.0, help="Match radius [arcsec].")
    p.set_defaults(func=make_truth)

    p = sub.add_parser("run", help="Fit the spectra and write metrics, plots and the comment.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dsps-ssp-fn", required=True)
    p.add_argument("--spectra", required=True, help="jaxsedfit chimera_validation_spectra.fits.")
    p.add_argument("--sample", required=True, help="jaxsedfit chimera_validation_sample.csv (truths).")
    p.add_argument("--truth", default=str(DEFAULT_TRUTH), help="Broad-line truth table.")
    p.add_argument("--chunk", type=int, default=16, help="Spectra per compiled batch.")
    p.add_argument("--steps", type=int, default=2000, help="MAP steps.")
    p.add_argument("--limit", type=int, default=None, help="Fit only the first N spectra (quick look).")
    p.add_argument("--sha", default="local")
    p.add_argument("--image-base-url", default=None, help="URL of the directory holding the plots.")
    p.set_defaults(func=run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
