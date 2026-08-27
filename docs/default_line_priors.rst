Emission-line prior tables
==========================

The publication-ready table containing the continuum, nuisance, and default
emission-line priors is available as
:download:`LaTeX source <prior_table.tex>`.

The tables below are generated directly from the built-in prior rows at
documentation-build time. Only the first table is enabled by default; the two
opt-in tables extend it for spectra that need more complete optical narrow-line
coverage.

Default broad-line AGN table
----------------------------

This is the line model activated by default when a complex overlaps the fitted
rest-frame wavelength range. Its values come from
:data:`jaxqsofit.defaults.DEFAULT_LINE_PRIOR_ROWS`.

.. default-line-table::

Optional optical and red/NIR narrow lines
-----------------------------------------

:data:`jaxqsofit.defaults.DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS` adds a denser
narrow-line list for emission-line-galaxy and host-dominated spectra. It
includes the resolved [O II] doublet, [Ne III], additional narrow Balmer lines,
He I, [O I], [N II], [S II], near-IR Paschen lines, and [S III]. Enable it when
constructing the prior configuration:

.. code-block:: python

   cfg.prior_config = PriorConfig.from_spectrum(
       flux=flux,
       redshift=z,
       include_elg_narrow_lines=True,
   )

.. optical-line-table::

Optional high-ionization and coronal lines
------------------------------------------

:data:`jaxqsofit.defaults.DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS` adds the
optical [Ne V], [Fe VII], and [Fe X] narrow lines. It can be enabled alone or
together with the broader optical table:

.. code-block:: python

   cfg.prior_config = PriorConfig.from_spectrum(
       flux=flux,
       redshift=z,
       include_high_ionization_lines=True,
   )

.. high-ionization-line-table::

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
