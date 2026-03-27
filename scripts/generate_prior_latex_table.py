#!/usr/bin/env python3
"""Generate a LaTeX prior-parameter table from JAXQSOFit source code.

This script is intentionally standalone. It loads ``defaults.py`` directly,
parses ``model.py`` for NumPyro sample sites, reconstructs the default tied-line
groups, and writes a LaTeX longtable describing the active Bayesian parameters.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "jaxqsofit"
DEFAULTS_PATH = SRC_DIR / "defaults.py"
MODEL_PATH = SRC_DIR / "model.py"

C_KMS = 299792.458


def _load_defaults_module():
    spec = importlib.util.spec_from_file_location("jaxqsofit_defaults", DEFAULTS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load defaults module from {DEFAULTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_literal_sample_sites(model_path: Path) -> set[str]:
    """Return literal string sample-site names in qso_fsps_joint_model."""
    tree = ast.parse(model_path.read_text())
    sites: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "numpyro"
                and node.func.attr == "sample"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                sites.add(node.args[0].value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return sites


def _append_unique_by_wavelength(
    base_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    atol_angstrom: float = 1.0,
) -> list[dict[str, Any]]:
    out = list(base_rows)
    for row in extra_rows:
        lam_new = float(row.get("lambda", np.nan))
        if not np.isfinite(lam_new):
            continue
        exists = False
        for old in out:
            lam_old = float(old.get("lambda", np.nan))
            if np.isfinite(lam_old) and abs(lam_old - lam_new) <= float(atol_angstrom):
                exists = True
                break
        if not exists:
            out.append(copy.deepcopy(row))
    return out


def _compress_group_ids(indices: list[int], compnames: list[str]) -> np.ndarray:
    group_ids: list[int] = []
    mapping: dict[tuple[str, int], int] = {}
    next_gid = 0
    for idx, comp in zip(indices, compnames):
        if idx < 0:
            group_ids.append(-1)
            continue
        key = (comp, idx)
        if key not in mapping:
            mapping[key] = next_gid
            next_gid += 1
        group_ids.append(mapping[key])
    return np.asarray(group_ids, dtype=int)


def _build_tied_line_groups(line_rows: list[dict[str, Any]], wave_min: float, wave_max: float) -> dict[str, Any]:
    rows = [row for row in line_rows if wave_min < float(row["lambda"]) < wave_max]

    names: list[str] = []
    compnames: list[str] = []
    vindex: list[int] = []
    windex: list[int] = []
    findex: list[int] = []
    fvalue: list[float] = []
    dmu_min: list[float] = []
    dmu_max: list[float] = []
    sig_init: list[float] = []
    sig_min: list[float] = []
    sig_max: list[float] = []
    amp_init: list[float] = []
    amp_min: list[float] = []
    amp_max: list[float] = []

    for row in rows:
        for i in range(int(row.get("ngauss", 1))):
            names.append(f"{row['linename']}_{i + 1}")
            compnames.append(str(row.get("compname", row["linename"])))
            vindex.append(int(row["vindex"]))
            windex.append(int(row["windex"]))
            findex.append(int(row["findex"]))
            fvalue.append(float(row["fvalue"]))
            dln = float(row["voff"]) / C_KMS
            dmu_min.append(-dln)
            dmu_max.append(+dln)
            sig_init.append(max(float(row["inisig"]), 1e-5))
            sig_min.append(max(float(row["minsig"]), 1e-5))
            sig_max.append(max(float(row["maxsig"]), 1e-5))
            amp_init.append(float(row["inisca"]))
            amp_min.append(float(row["minsca"]))
            amp_max.append(float(row["maxsca"]))

    vgroup = _compress_group_ids(vindex, compnames)
    wgroup = _compress_group_ids(windex, compnames)
    fgroup = _compress_group_ids(findex, compnames)

    flux_ratio = np.ones(len(fgroup), dtype=float)
    for gid in sorted({g for g in fgroup if g >= 0}):
        members = np.where(fgroup == gid)[0]
        ref = members[0]
        ref_f = fvalue[ref] if fvalue[ref] != 0 else 1.0
        for m in members:
            flux_ratio[m] = fvalue[m] / ref_f if ref_f != 0 else 1.0

    n_vgroups = int(np.max(vgroup)) + 1 if len(vgroup) else 0
    n_wgroups = int(np.max(wgroup)) + 1 if len(wgroup) else 0
    n_fgroups = int(np.max(fgroup)) + 1 if len(fgroup) else 0

    dmu_min = np.asarray(dmu_min, dtype=float)
    dmu_max = np.asarray(dmu_max, dtype=float)
    sig_init = np.asarray(sig_init, dtype=float)
    sig_min = np.asarray(sig_min, dtype=float)
    sig_max = np.asarray(sig_max, dtype=float)
    amp_init = np.asarray(amp_init, dtype=float)
    amp_min = np.asarray(amp_min, dtype=float)
    amp_max = np.asarray(amp_max, dtype=float)

    dmu_groups = []
    for gid in range(n_vgroups):
        members = np.where(vgroup == gid)[0]
        dmu_groups.append(
            {
                "gid": gid,
                "members": [names[i] for i in members],
                "init": 0.0,
                "min": float(np.max(dmu_min[members])),
                "max": float(np.min(dmu_max[members])),
            }
        )

    sig_groups = []
    for gid in range(n_wgroups):
        members = np.where(wgroup == gid)[0]
        sig_groups.append(
            {
                "gid": gid,
                "members": [names[i] for i in members],
                "init": float(np.median(sig_init[members])),
                "min": float(np.max(sig_min[members])),
                "max": float(np.min(sig_max[members])),
            }
        )

    amp_groups = []
    for gid in range(n_fgroups):
        members = np.where(fgroup == gid)[0]
        ref = members[0]
        amp_groups.append(
            {
                "gid": gid,
                "members": [names[i] for i in members],
                "ratios": [float(flux_ratio[i]) for i in members],
                "init": float(amp_init[ref]),
                "min": float(amp_min[ref]),
                "max": float(amp_max[ref]),
            }
        )

    return {
        "dmu_groups": dmu_groups,
        "sig_groups": sig_groups,
        "amp_groups": amp_groups,
    }


def _fmt_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e3 or ax < 1e-3:
        s = f"{x:.1e}"
        mant, exp = s.split("e")
        return f"{mant}\\times10^{{{int(exp)}}}"
    return f"{x:.3g}"


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _dist_label(cfg: dict[str, Any]) -> str:
    dist_name = str(cfg.get("dist", "Unknown"))
    if dist_name.lower() == "normal":
        return f"$\\mathcal{{N}}({_fmt_float(cfg['loc'])},\\ {_fmt_float(cfg['scale'])})$"
    if dist_name.lower() == "lognormal":
        return f"$\\log\\,\\mathcal{{N}}({_fmt_float(cfg['loc'])},\\ {_fmt_float(cfg['scale'])})$"
    if dist_name.lower() == "halfnormal":
        if "scale" in cfg:
            return f"$\\mathcal{{HN}}({_fmt_float(cfg['scale'])})$"
        if "scale_mult_err" in cfg:
            return f"$\\mathcal{{HN}}({_fmt_float(cfg['scale_mult_err'])}\\langle\\mathrm{{err}}\\rangle)$"
    if dist_name.lower() == "studentt":
        return (
            "$\\mathcal{T}("
            f"{_fmt_float(cfg['df'])},\\ {_fmt_float(cfg['loc'])},\\ {_fmt_float(cfg['scale'])})$"
        )
    if dist_name.lower() == "truncatednormal":
        return (
            "$\\mathcal{TN}("
            f"{_fmt_float(cfg['loc'])},\\ {_fmt_float(cfg['scale'])},\\ "
            f"[{_fmt_float(cfg['low'])},{_fmt_float(cfg['high'])}])$"
        )
    return _latex_escape(repr(cfg))


def _build_main_rows(
    prior: dict[str, Any], sample_sites: set[str], fit_poly_order: int
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    rows: list[tuple[str, str, str]] = []
    line_family_rows: list[tuple[str, str, str]] = []

    def add(name: str, dist_text: str, desc: str) -> None:
        rows.append((f"\\texttt{{{_latex_escape(name)}}}", dist_text, desc))

    def add_line_family(name: str, dist_text: str, desc: str) -> None:
        line_family_rows.append((f"\\texttt{{{_latex_escape(name)}}}", dist_text, desc))

    if "cont_norm" in sample_sites:
        add("cont_norm", _dist_label(prior["log_cont_norm"]), "Total continuum normalization before splitting into AGN and host components.")
    if "log_frac_host" in sample_sites:
        add("log_frac_host", _dist_label(prior["log_frac_host"]), "Latent host-fraction parameter, transformed with a sigmoid to give $f_{\\rm host}\\in(0,1)$.")
    if "PL_norm" in sample_sites:
        add("PL_norm", _dist_label(prior["PL_norm"]), "AGN power-law continuum amplitude.")
    if "PL_slope" in sample_sites:
        add("PL_slope", _dist_label(prior["PL_slope"]), "AGN power-law spectral slope.")
    if "reddening_ebv" in sample_sites:
        add("reddening_ebv", _dist_label(prior["reddening_ebv"]), "SMC-like AGN reddening amplitude.")

    if "Fe_uv_norm" in sample_sites:
        add("Fe_uv_norm", _dist_label(prior["log_Fe_uv_norm"]), "UV Fe template normalization.")
    if "log_Fe_op_over_uv" in sample_sites:
        add("log_Fe_op_over_uv", _dist_label(prior["log_Fe_op_over_uv"]), "Log optical-to-UV Fe normalization ratio.")
    if "Fe_uv_FWHM" in sample_sites:
        add("Fe_uv_FWHM", _dist_label(prior["log_Fe_uv_FWHM"]), "UV Fe template broadening FWHM in km s$^{-1}$.")
    if "Fe_op_FWHM" in sample_sites:
        add("Fe_op_FWHM", _dist_label(prior["log_Fe_op_FWHM"]), "Optical Fe template broadening FWHM in km s$^{-1}$.")
    if "Fe_uv_shift" in sample_sites:
        add("Fe_uv_shift", _dist_label(prior["Fe_uv_shift"]), "UV Fe template shift in log-wavelength units.")
    if "Fe_op_shift" in sample_sites:
        add("Fe_op_shift", _dist_label(prior["Fe_op_shift"]), "Optical Fe template shift in log-wavelength units.")

    if "Balmer_norm" in sample_sites:
        add("Balmer_norm", _dist_label(prior["log_Balmer_norm"]), "Balmer continuum normalization.")
    if "Balmer_Tau" in sample_sites:
        add("Balmer_Tau", _dist_label(prior["log_Balmer_Tau"]), "Balmer continuum optical depth.")
    if "Balmer_vel" in sample_sites:
        add("Balmer_vel", _dist_label(prior["log_Balmer_vel"]), "Balmer continuum velocity broadening in km s$^{-1}$.")

    if "tau_host" in sample_sites:
        add("tau_host", _dist_label(prior["tau_host"]), "Hierarchical scale controlling the spread of raw FSPS template weights.")
    if "fsps_weights_raw" in sample_sites:
        raw = prior["raw_w"]
        dist_text = f"$\\mathcal{{N}}({_fmt_float(raw['loc'])},\\ \\tau_{{\\rm host}})$"
        add("fsps_weights_raw[$i$]", dist_text, "Raw latent FSPS template weight for template $i$ before softmax normalization.")
    if "gal_v_kms" in sample_sites:
        add("gal_v_kms", _dist_label(prior["gal_v_kms"]), "Host stellar velocity offset in km s$^{-1}$.")
    if "gal_sigma_kms" in sample_sites:
        add("gal_sigma_kms", _dist_label(prior["gal_sigma_kms"]), "Host stellar velocity dispersion in km s$^{-1}$.")

    if "line_dmu_group" in sample_sites:
        dmu_dist = (
            "$\\mathcal{TN}(d\\mu_{\\rm init,g},\\ "
            "\\texttt{line\\_dmu\\_scale\\_mult}\\,[d\\mu_{\\max,g}-d\\mu_{\\min,g}],\\ "
            "[d\\mu_{\\min,g},d\\mu_{\\max,g}])$"
        )
        add_line_family("line_dmu_group[$g$]", dmu_dist, "Velocity-offset parameter for tied line group $g$ in log-wavelength units.")
    if "line_sig_group" in sample_sites:
        sig_dist = (
            "$\\mathcal{TN}(\\sigma_{\\rm init,g},\\ "
            "\\texttt{line\\_sig\\_scale\\_mult}\\,[\\sigma_{\\max,g}-\\sigma_{\\min,g}],\\ "
            "[\\sigma_{\\min,g},\\sigma_{\\max,g}])$"
        )
        add_line_family("line_sig_group[$g$]", sig_dist, "Width parameter for tied line group $g$.")
    if "line_amp_group" in sample_sites:
        amp_dist = (
            "$\\mathcal{TN}(A_{\\rm init,g},\\ "
            "\\texttt{line\\_amp\\_scale\\_mult}\\,[A_{\\max,g}-A_{\\min,g}],\\ "
            "[A_{\\min,g},A_{\\max,g}])$"
        )
        add_line_family("line_amp_group[$g$]", amp_dist, "Base amplitude for tied flux group $g$; per-component amplitudes are fixed-ratio scalings of this group amplitude.")

    if fit_poly_order > 0:
        for k in range(1, fit_poly_order + 1):
            key = f"poly_c{k}"
            if key in prior:
                add(f"poly_c{k}", _dist_label(prior[key]), f"Coefficient of the multiplicative continuum tilt polynomial term of order {k}.")

    if "frac_jitter" in sample_sites:
        add("frac_jitter", _dist_label(prior["frac_jitter"]), "Fractional model-dependent extra scatter term.")
    if "add_jitter" in sample_sites:
        add("add_jitter", _dist_label(prior["add_jitter"]), "Additive extra scatter term.")
    if "delta_m_psf_raw" in sample_sites:
        add("delta_m_psf_raw", "$\\mathcal{N}(0, 0.5)$", "Global magnitude offset for the PSF-photometry forward model.")
    if "eta_psf_raw" in sample_sites:
        add("eta_psf_raw", "$\\mathrm{Beta}(2, 2)$", "Host suppression factor in the PSF-photometry forward model.")
    if "sigma_phot_extra" in sample_sites:
        add("sigma_phot_extra", "$\\mathcal{HN}(0.05)$", "Extra photometric scatter for PSF magnitude constraints.")

    return rows, line_family_rows


def _build_line_group_rows(groups: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    line_label_map = {
        "Ha": r"H$\alpha$",
        "Hb": r"H$\beta$",
        "Hg": r"H$\gamma$",
        "Hd": r"H$\delta$",
        "Lya": r"Ly$\alpha$",
        "MgII": "Mg II",
        "CIII": "C III]",
        "CIV": "C IV",
        "OII": "[O II]",
        "OIII": "[O III]",
        "NII": "[N II]",
        "SII": "[S II]",
        "NeV": "[Ne V]",
        "SiIV": "Si IV",
        "NV": "N V",
    }

    def pretty_group_name(members: list[str], prefix: str) -> str:
        tokens: list[str] = []
        broad = False
        narrow = False
        core = False
        wing = False
        for member in members:
            base = member.rsplit("_", 1)[0]
            broad = broad or "_br" in base
            narrow = narrow or "_na" in base
            core = core or base.endswith("c")
            wing = wing or base.endswith("w")
            root = base
            for suffix in ("_br", "_na", "c", "w"):
                if root.endswith(suffix):
                    root = root[: -len(suffix)]
                    break
            root = root.split("_")[0]
            label = line_label_map.get(root, root)
            if label not in tokens:
                tokens.append(label)
        joined = " + ".join(tokens) if tokens else "lines"
        qualifiers: list[str] = []
        if broad and not (narrow or core or wing):
            qualifiers.append("broad")
        elif narrow and not (broad or core or wing):
            qualifiers.append("narrow")
        else:
            if broad:
                qualifiers.append("broad")
            if narrow:
                qualifiers.append("narrow")
            if core:
                qualifiers.append("core")
            if wing:
                qualifiers.append("wing")
        if qualifiers:
            joined = f"{joined}; " + "/".join(qualifiers)
        return f"\\texttt{{{_latex_escape(prefix)}}} ({joined})"

    for g in groups["amp_groups"]:
        span = max(g["max"] - g["min"], 0.0)
        dist_text = (
            "$\\mathcal{TN}("
            f"{_fmt_float(g['init'])},\\ 0.25\\times{_fmt_float(span)},\\ "
            f"[{_fmt_float(g['min'])},{_fmt_float(g['max'])}])$"
        )
        ratios = ", ".join(_fmt_float(v) for v in g["ratios"])
        members = ", ".join(f"\\texttt{{{_latex_escape(m)}}}" for m in g["members"])
        desc = f"Tied flux-amplitude group with members {members}; fixed flux ratios ({ratios})."
        rows.append((pretty_group_name(g["members"], "line_amp"), dist_text, desc))

    sig_entries: list[tuple[str, str]] = []
    for g in groups["sig_groups"]:
        span = max(g["max"] - g["min"], 0.0)
        dist_text = (
            "$\\mathcal{TN}("
            f"{_fmt_float(g['init'])},\\ 0.25\\times{_fmt_float(span)},\\ "
            f"[{_fmt_float(g['min'])},{_fmt_float(g['max'])}])$"
        )
        sig_entries.append((pretty_group_name(g["members"], "line_sig"), dist_text))

    dmu_entries: list[tuple[str, str]] = []
    for g in groups["dmu_groups"]:
        span = max(g["max"] - g["min"], 0.0)
        dist_text = (
            "$\\mathcal{TN}("
            f"{_fmt_float(g['init'])},\\ 0.25\\times{_fmt_float(span)},\\ "
            f"[{_fmt_float(g['min'])},{_fmt_float(g['max'])}])$"
        )
        dmu_entries.append((pretty_group_name(g["members"], "line_dmu"), dist_text))

    def aggregate_shared(
        entries: list[tuple[str, str]],
        shared_name: str,
        quantity_desc: str,
    ) -> list[tuple[str, str, str]]:
        grouped: dict[str, list[str]] = {}
        for label, dist_text in entries:
            grouped.setdefault(dist_text, []).append(label)
        out: list[tuple[str, str, str]] = []
        for dist_text, labels in grouped.items():
            if len(labels) == 1:
                out.append((labels[0], dist_text, f"{quantity_desc} applied to {labels[0]}."))
            else:
                joined = ", ".join(labels)
                out.append(
                    (
                        f"\\texttt{{{_latex_escape(shared_name)}}}",
                        dist_text,
                        f"Shared {quantity_desc.lower()} applied to {joined}.",
                    )
                )
        return out

    rows.extend(
        aggregate_shared(
            sig_entries,
            shared_name="line_sig_shared",
            quantity_desc="line-width prior",
        )
    )
    rows.extend(
        aggregate_shared(
            dmu_entries,
            shared_name="line_dmu_shared",
            quantity_desc="velocity-offset prior",
        )
    )

    return rows


def _render_table(rows: list[tuple[str, str, str]], caption: str) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{p{0.22\\linewidth} p{0.30\\linewidth} p{0.42\\linewidth}}",
        "\\hline",
        "\\textbf{Parameter} & \\textbf{Prior Distribution} & \\textbf{Parameter Description} \\\\",
        "\\hline",
    ]
    for name, dist_text, desc in rows:
        lines.append(f"{name} & {dist_text} & {desc} \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\vspace{0.4em}",
        "\\begin{minipage}{0.94\\linewidth}",
        "\\footnotesize",
        "\\textit{Note.} "
        "$\\mathcal{N}(\\mu,\\sigma)$ indicates a normal prior with mean $\\mu$ and standard deviation $\\sigma$. "
        "$\\mathcal{HN}(\\sigma)$ indicates a half-normal prior with scale $\\sigma$. "
        "$\\log\\,\\mathcal{N}(\\mu,\\sigma)$ indicates a log-normal prior whose logarithm is normally distributed with mean $\\mu$ and standard deviation $\\sigma$. "
        "$\\mathcal{TN}(\\mu,\\sigma,[a,b])$ indicates a truncated normal prior with location $\\mu$, scale $\\sigma$, "
        "lower bound $a$, and upper bound $b$. "
        "$\\mathcal{T}(\\nu,\\mu,\\sigma)$ indicates a Student-$t$ prior with degrees of freedom $\\nu$, "
        "location $\\mu$, and scale $\\sigma$. "
        "$\\mathcal{U}(a,b)$ indicates a uniform prior between the minimum value $a$ and maximum value $b$.",
        "\\end{minipage}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _render_tables(
    main_rows: list[tuple[str, str, str]],
    line_rows: list[tuple[str, str, str]],
    main_caption: str,
    line_caption: str,
) -> str:
    blocks = [_render_table(main_rows, main_caption)]
    if line_rows:
        blocks.append("")
        blocks.append(_render_table(line_rows, line_caption))
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional output .tex file. Defaults to stdout.")
    parser.add_argument("--flux-scale", type=float, default=1.0, help="Representative flux scale used to build default priors.")
    parser.add_argument("--wave-min", type=float, default=1150.0, help="Minimum rest wavelength for active line groups.")
    parser.add_argument("--wave-max", type=float, default=6800.0, help="Maximum rest wavelength for active line groups.")
    parser.add_argument("--fit-poly-order", type=int, default=2, help="Polynomial order to include in the table.")
    parser.add_argument("--include-elg-narrow-lines", action="store_true", help="Append default ELG narrow lines before grouping.")
    parser.add_argument("--include-high-ionization-lines", action="store_true", help="Append default high-ionization lines before grouping.")
    parser.add_argument("--include-line-groups", action="store_true", help="Include one row per default tied-line group.")
    args = parser.parse_args()

    defaults = _load_defaults_module()
    sample_sites = _extract_literal_sample_sites(MODEL_PATH)

    flux = np.full(128, float(args.flux_scale), dtype=float)
    prior = defaults.build_default_prior_config(
        flux=flux,
        include_elg_narrow_lines=args.include_elg_narrow_lines,
        include_high_ionization_lines=args.include_high_ionization_lines,
    )

    line_rows = copy.deepcopy(prior["line"]["table"])
    if args.include_elg_narrow_lines:
        line_rows = _append_unique_by_wavelength(
            line_rows,
            copy.deepcopy(defaults.DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS),
            atol_angstrom=1.0,
        )
    if args.include_high_ionization_lines:
        line_rows = _append_unique_by_wavelength(
            line_rows,
            copy.deepcopy(defaults.DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS),
            atol_angstrom=1.0,
        )

    main_rows, line_family_rows = _build_main_rows(prior=prior, sample_sites=sample_sites, fit_poly_order=args.fit_poly_order)
    line_group_rows: list[tuple[str, str, str]] = list(line_family_rows)
    if args.include_line_groups:
        groups = _build_tied_line_groups(line_rows, wave_min=args.wave_min, wave_max=args.wave_max)
        line_group_rows = _build_line_group_rows(groups)

    main_caption = "JAXQSOFit prior parameters generated from \\texttt{defaults.py} and \\texttt{model.py}."
    line_caption = "JAXQSOFit tied emission-line prior groups generated from the default line configuration."
    latex = _render_tables(main_rows, line_group_rows, main_caption, line_caption)

    if args.output is not None:
        args.output.write_text(latex + "\n")
    else:
        print(latex)


if __name__ == "__main__":
    main()
