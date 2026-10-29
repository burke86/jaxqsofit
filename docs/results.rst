Working with results
====================

:meth:`jaxqsofit.JAXQSOFit.fit` returns a :class:`jaxqsofit.FitResult`, the same
for every inference method (see :doc:`inference`). ``result.samples`` holds the
parameter draws, and ``result.spectrum`` the fitted spectrum and line
measurements.

.. _reading-spectral-results:

Spectra and lines
-----------------

``result.spectrum`` is the supported interface for spectral results. It has named,
unit-explicit fields, so user code does not depend on internal NumPyro site names.
Every numerical field keeps the posterior-draw axis.

Each Gaussian component of a line has its own name:

.. code-block:: python

   result = q.fit()

   hb1 = result.spectrum.lines["Hb_br_1"]
   hb1.amplitude_flambda_1e17, hb1.fwhm_kms, hb1.flux_erg_s_cm2

The available fields are ``amplitude_flambda_1e17`` (in
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}`), ``center_rest_angstrom``,
``sigma_ln_lambda``, ``fwhm_kms``, ``velocity_offset_kms`` and ``flux_erg_s_cm2``
(in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`). The metadata fields are ``parent_line``,
``component_index``, ``kind`` and ``rest_wavelength_angstrom``.

Single-component lines have no ``_1`` suffix: the model's ``OIII_5007_1`` is
``result.spectrum.lines["OIII_5007"]``. To sum the components of a physical line,
use ``line_groups``:

.. code-block:: python

   group = result.spectrum.line_groups["Hb_br"]
   group.component_names                  # ("Hb_br_1", "Hb_br_2", ...)
   hb_total_flux_draws = group.total_flux_erg_s_cm2

The fitted spectrum and its components have shape ``(draw, pixel)``; the wavelength,
observed flux, error and mask have shape ``(pixel,)``:

.. code-block:: python

   spectrum = result.spectrum
   wave = spectrum.wavelength_rest_angstrom
   observed = spectrum.observed_flux_flambda_1e17
   error = spectrum.error_flambda_1e17
   model_draws = spectrum.model_flambda_1e17
   continuum_draws = spectrum.continuum_flambda_1e17
   line_draws = spectrum.line_flambda_1e17
   feii_draws = spectrum.feii_flambda_1e17
   balmer_draws = spectrum.balmer_continuum_flambda_1e17
   host_draws = spectrum.host_flambda_1e17
   power_law_draws = spectrum.power_law_flambda_1e17

``result.predict(n_draws=200).spectrum`` reconstructs the spectrum with fewer
draws, or on another wavelength grid. On a grid other than the data's, the observed
flux and error are ``NaN`` and the mask is false, because no observed pixel
corresponds to each model pixel.

Luminosities
------------

Line luminosities with uncertainties follow from the flux draws, for example for
the full broad H-beta line:

.. code-block:: python

   import numpy as np
   from astropy.cosmology import FlatLambdaCDM

   cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

   flux_draws = result.spectrum.line_groups["Hb_br"].total_flux_erg_s_cm2
   d_l_cm = cosmo.luminosity_distance(q.z).to("cm").value
   luminosity_draws = flux_draws * 4.0 * np.pi * d_l_cm**2
   np.nanpercentile(np.log10(luminosity_draws), [16, 50, 84])

Line summary table
------------------

After a fit, ``q.line_result`` holds the posterior median of every Gaussian
component's parameters and ``q.line_result_name`` the matching names:

.. code-block:: python

   line = dict(zip(q.line_result_name, q.line_result))

   amplitude = line["Hb_br_1_scale"]
   center_lnlam = line["Hb_br_1_centerwave"]
   sigma_lnlam = line["Hb_br_1_sigma"]
   sigma_lnlam_err = line["Hb_br_1_sigma_err"]

   center_angstrom = np.exp(center_lnlam)
   velocity_kms = 299792.458 * (center_lnlam - np.log(4862.68))

Each component has ``_scale``, ``_centerwave`` and ``_sigma`` fields with ``_err``
counterparts: the medians and the posterior standard deviations. Despite its name,
``_centerwave`` is the natural log of the rest wavelength in Å, and ``_sigma`` is a
width in :math:`\ln\lambda`, so the velocity offset from a laboratory wavelength
:math:`\lambda_0` is :math:`c\,(\ln\lambda - \ln\lambda_0)` and the velocity
dispersion is about :math:`c\,\sigma_{\ln\lambda}`.

With ``apply_instrumental_resolution=True``, ``_sigma`` is the intrinsic,
resolution-corrected width; otherwise it is the observed, instrument-broadened
width (see :doc:`model`). The widths used to generate the model are in
``q.pred_out["line_sig_effective_per_component"]``.

These are compact summaries. For credible intervals, covariances or velocity
separations between components, use the posterior draws in ``result.spectrum``.

Plots
-----

.. code-block:: python

   q.plot_fig()                          # data, components, line decomposition, residuals
   result.plot_corner(show_plot=False)
   result.plot_trace()

Saving and loading
------------------

With ``cfg.output.save_result = True``, or by calling
:meth:`jaxqsofit.FitResult.save`, a fit is written to an HDF5 bundle named
``<object_id>_samples.h5``:

.. code-block:: python

   result.save("fit_outputs")
   loaded = JAXQSOFit.load_result("fit_outputs/demo_samples.h5")

:func:`jaxqsofit.load_from_samples` reloads a bundle into a fitter, optionally
replotting it.

Lower-level attributes
----------------------

The fitter also keeps arrays from the last fit. They are useful for diagnostics,
but their names are not a stable interface:

* the data: ``wave``, ``flux``, ``err``;
* the posterior-median model: ``model_total``, ``f_conti_model``, ``f_line_model``,
  ``f_pl_model``, ``f_fe_mgii_model``, ``f_fe_balmer_model``, ``f_bc_model``,
  ``host``, ``qso``, and ``custom_components`` (user-defined and BAL components);
* host and Balmer-continuum fractions: ``frac_host_4200``, ``frac_host_5100``,
  ``frac_host_2500`` and ``frac_bc_2500``;
* the posterior: ``numpyro_mcmc``, ``numpyro_samples`` and ``pred_out``.
