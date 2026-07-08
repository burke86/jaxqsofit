Tutorials
=========

Notebook tutorials
------------------

The notebooks below are rendered into the documentation with ``nbsphinx``.
By default, docs builds execute the notebooks so Read the Docs and CI render
fresh outputs. To skip execution during a local text-only docs edit, set
``NBSPHINX_EXECUTE=never`` before invoking Sphinx.

.. toctree::
   :maxdepth: 1

   notebooks/01_jaxqsofit_tutorial
   notebooks/02_selsing_composite_fit
   notebooks/03_sdss_j102839_iron
   notebooks/04_sdss_galaxy
   notebooks/05_sdss_psf_recalibration_tutorial
   notebooks/06_custom_components_tutorial
   notebooks/07_local_sdss_bal_tutorial

What this tutorial covers
-------------------------

- Fetching an SDSS spectrum near ``(ra, dec) = (184.0307, -2.2383)``
- Running ``JAXQSOFit`` with ``nuts``, ``optax``, and ``optax+nuts``
- Overriding priors via ``prior_config``
- Computing broad-line FWHM and luminosity from fitted components
