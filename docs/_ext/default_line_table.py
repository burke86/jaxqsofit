"""Sphinx directive rendering the active JAXQSOFit emission-line defaults."""

from __future__ import annotations

from collections import OrderedDict
from html import escape

from docutils import nodes
from docutils.parsers.rst import Directive

_C_KMS = 299_792.458

_WIDTH_LABELS = {
    (0.005, 0.004, 0.05): "Broad",
    (0.001, 0.00023, 0.00169): "Narrow",
    (0.001, 0.0005, 0.00169): "Relaxed narrow",
    (0.001, 0.0003333, 0.00169): "UV narrow",
    (0.003, 0.00023, 0.004): "[O III] wing",
    (0.005, 0.002, 0.05): "UV broad",
    (0.002, 0.001, 0.01): "Intermediate UV",
    (0.001, 0.0005, 0.002): "Relaxed UV narrow",
    (0.002, 0.001, 0.015): "Extended UV",
    (0.005, 0.001, 0.05): "C IV broad",
    (0.005, 0.0025, 0.02): "UV semibroad",
}


def _number(value: float, digits: int = 0) -> str:
    """Format a table value compactly with thousands separators."""
    return f"{value:,.{digits}f}"


def _width_label(row: dict) -> str:
    key = tuple(round(float(row[field]), 7) for field in ("inisig", "minsig", "maxsig"))
    return _WIDTH_LABELS.get(key, "Custom")


def _line_kind(name: str) -> str:
    if "_br" in name:
        return "broad"
    if name.endswith("w"):
        return "wing"
    return "narrow"


def _badge(prefix: str, index: int) -> str:
    if index <= 0:
        return '<span class="line-tie independent">—</span>'
    return (
        f'<span class="line-tie tie-{prefix.lower()}" '
        f'title="{escape(prefix)} tie group {index}">{escape(prefix)}{index}</span>'
    )


def _row_html(row: dict) -> str:
    name = str(row["linename"])
    kind = _line_kind(name)
    sigma_initial = float(row["inisig"]) * _C_KMS
    sigma_min = float(row["minsig"]) * _C_KMS
    sigma_max = float(row["maxsig"]) * _C_KMS
    shift = float(row["voff"]) * _C_KMS
    findex = int(row["findex"])
    amplitude = float(row["fvalue"])

    if findex > 0:
        amplitude_html = (
            f'{_badge("A", findex)}'
            f'<span class="amplitude-factor">× {_number(amplitude, 3)}</span>'
        )
    else:
        amplitude_html = (
            '<span class="line-tie independent">—</span>'
            f'<span class="amplitude-factor">init {_number(amplitude, 4)}</span>'
        )

    return f"""
      <tr>
        <td class="line-name">
          <code>{escape(name)}</code>
          <span class="line-kind kind-{kind}">{kind}</span>
        </td>
        <td class="numeric">{float(row["lambda"]):,.2f}</td>
        <td>
          <span class="width-profile">{escape(_width_label(row))}</span>
          <small>{_number(sigma_initial)} [{_number(sigma_min)}–{_number(sigma_max)}]</small>
        </td>
        <td class="numeric">±{_number(shift)}</td>
        <td class="tie-cell">{_badge("V", int(row["vindex"]))}</td>
        <td class="tie-cell">{_badge("W", int(row["windex"]))}</td>
        <td class="amplitude-cell">{amplitude_html}</td>
        <td class="numeric gaussian-count">{int(row["ngauss"])}</td>
      </tr>"""


def _complex_html(name: str, rows: list[dict]) -> str:
    body = "".join(_row_html(row) for row in rows)
    wavelength_min = min(float(row["lambda"]) for row in rows)
    wavelength_max = max(float(row["lambda"]) for row in rows)
    wavelength_note = (
        f"{wavelength_min:,.0f} Å"
        if wavelength_min == wavelength_max
        else f"{wavelength_min:,.0f}–{wavelength_max:,.0f} Å"
    )
    return f"""
    <section class="line-complex-card">
      <header>
        <div>
          <span class="complex-kicker">Fitting complex</span>
          <h3>{escape(name)}</h3>
        </div>
        <span class="complex-meta">{len(rows)} definitions · {wavelength_note}</span>
      </header>
      <div class="line-table-scroll">
        <table class="default-line-table">
          <thead>
            <tr>
              <th>Line</th>
              <th>λ<sub>vac</sub> [Å]</th>
              <th>σ<sub>v</sub> prior: init [range] [km s<sup>−1</sup>]</th>
              <th>Center shift [km s<sup>−1</sup>]</th>
              <th>Velocity</th>
              <th>Width</th>
              <th>Amplitude</th>
              <th>Gaussians</th>
            </tr>
          </thead>
          <tbody>{body}
          </tbody>
        </table>
      </div>
    </section>"""


def _table_html(rows: list[dict]) -> str:
    complexes: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        complexes.setdefault(str(row["compname"]), []).append(row)

    gaussian_count = sum(int(row["ngauss"]) for row in rows)
    broad_count = sum("_br" in str(row["linename"]) for row in rows)
    cards = "".join(_complex_html(name, grouped) for name, grouped in complexes.items())
    return f"""
<div class="line-prior-document">
  <div class="line-prior-summary">
    <div><strong>{len(rows)}</strong><span>line definitions</span></div>
    <div><strong>{gaussian_count}</strong><span>Gaussian components</span></div>
    <div><strong>{len(complexes)}</strong><span>fitting complexes</span></div>
    <div><strong>{broad_count}</strong><span>broad-line definitions</span></div>
  </div>
  <div class="line-prior-legend" aria-label="Line type legend">
    <span class="line-kind kind-broad">broad</span>
    <span class="line-kind kind-narrow">narrow</span>
    <span class="line-kind kind-wing">wing</span>
    <span class="legend-copy">σ values are shown in velocity units</span>
  </div>
  {cards}
</div>"""


class DefaultLineTableDirective(Directive):
    """Insert the current default emission-line table."""

    has_content = False

    def run(self):
        from jaxqsofit.defaults import DEFAULT_LINE_PRIOR_ROWS

        html = _table_html([dict(row) for row in DEFAULT_LINE_PRIOR_ROWS])
        return [nodes.raw("", html, format="html")]


def setup(app):
    """Register the directive with Sphinx."""
    app.add_directive("default-line-table", DefaultLineTableDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
