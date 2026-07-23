Default emission-line priors
============================

The publication-ready table containing the continuum, nuisance, and default
emission-line priors is available as
:download:`LaTeX source <prior_table.tex>`.

This is the line model activated by default when a complex overlaps the fitted
rest-frame wavelength range. Values are generated directly from
``DEFAULT_LINE_PRIOR_ROWS`` at documentation-build time.

.. default-line-table::

How to read the table
---------------------

Widths are Gaussian :math:`\sigma_v` values converted from the model's
ln-wavelength coordinate using :math:`v \simeq c\,\sigma_{\ln\lambda}`.
The center-shift column gives the symmetric allowed range around the laboratory
wavelength. These velocity conversions are intended to make the priors easy to
interpret; the model continues to evaluate profiles in ln-wavelength.

The ``v/w/f`` column lists the velocity, width, and amplitude tie indices.
Tie indices are scoped to a single fitting complex:

- matching positive ``v`` indices share a velocity offset;
- matching positive ``w`` indices share a Gaussian width;
- matching positive ``f`` indices share an amplitude coordinate.

Zero means the corresponding coordinate is independent. For tied amplitudes,
the displayed :math:`A_0` value is the row's fixed peak-amplitude factor.
The ratio helper in the defaults converts physically specified integrated-flux
ratios into the appropriate ln-wavelength peak ratios.

Rows with more than one Gaussian expand into independently parameterized
components. Their broad widths are ordered during sampling to remove
label-switched posterior modes.
