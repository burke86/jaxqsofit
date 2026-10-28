Tutorials
=========

Notebook tutorials
------------------

The notebooks below are rendered into the documentation with ``nbsphinx``.
Documentation builds render their saved outputs without executing them. Set
``NBSPHINX_EXECUTE=always`` explicitly to rerun notebooks during a local build.

.. toctree::
   :maxdepth: 1

   notebooks/01_jaxqsofit_tutorial
   notebooks/02_selsing_composite_fit
   notebooks/03_sdss_j102839_iron
   notebooks/04_sdss_galaxy
   notebooks/05_sdss_psf_recalibration_tutorial
   notebooks/06_custom_components_tutorial
   notebooks/07_local_sdss_bal_tutorial

What the first tutorial covers
------------------------------

- Downloading an SDSS spectrum with ``astroquery``
- Overriding the default priors with ``PriorConfig``
- Configuring a fit with ``FitConfig`` and choosing ``optax``, ``optax+nuts``, or ``nuts``
- Inspecting the posterior and the fitted components through ``result.spectrum``
- Measuring broad-line FWHM and luminosity with posterior uncertainties
