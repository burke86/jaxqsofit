"""Sphinx directive rendering the active JAXQSOFit emission-line defaults."""

from __future__ import annotations

from html import escape

from docutils import nodes
from docutils.parsers.rst import Directive

_C_KMS = 299_792.458


def _velocity(value: float) -> str:
    """Convert a width in ln-wavelength to a compact velocity value."""
    return f"{float(value) * _C_KMS:,.0f}"


def _amplitude(value: float) -> str:
    """Format an initial or relative peak amplitude."""
    return f"{float(value):.4g}"


def _tie_indices(row: dict) -> str:
    """Format velocity, width, and amplitude tie indices."""
    return "/".join(str(int(row[key])) for key in ("vindex", "windex", "findex"))


def _row_html(row: dict) -> str:
    return f"""
      <tr>
        <td><code>{escape(str(row["linename"]))}</code></td>
        <td>{escape(str(row["compname"]))}</td>
        <td class="numeric">{float(row["lambda"]):,.2f}</td>
        <td class="numeric centered">{int(row["ngauss"])}</td>
        <td class="numeric centered">{_tie_indices(row)}</td>
        <td class="numeric">{_amplitude(row["fvalue"])}</td>
        <td class="numeric">{_velocity(row["inisig"])}
          [{_velocity(row["minsig"])}–{_velocity(row["maxsig"])}]</td>
        <td class="numeric">±{_velocity(row["voff"])}</td>
      </tr>"""


def _table_html(rows: list[dict]) -> str:
    body = "".join(_row_html(row) for row in rows)
    return f"""
<div class="line-prior-document">
  <div class="line-table-scroll">
    <table class="default-line-table">
      <thead>
        <tr>
          <th>Line</th>
          <th>Complex</th>
          <th>λ<sub>vac</sub> [Å]</th>
          <th><i>N</i><sub>G</sub></th>
          <th>ties <i>v/w/f</i></th>
          <th><i>A</i><sub>0</sub></th>
          <th>σ<sub>v</sub>: initial [range] [km s<sup>−1</sup>]</th>
          <th>center shift [km s<sup>−1</sup>]</th>
        </tr>
      </thead>
      <tbody>{body}
      </tbody>
    </table>
  </div>
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
