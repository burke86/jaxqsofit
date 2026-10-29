JAXQSOFit Documentation
=======================

JAXQSOFit is a Bayesian quasar spectral fitting package built with JAX and
NumPyro. It is a JAX port of the PyQSOFit decomposition: a power-law AGN
continuum, Fe II and Balmer-continuum emission, tied Gaussian emission lines and a
host galaxy, fitted jointly in one probabilistic model rather than in stages. Fits
return posterior draws of every component and line measurement, from a fast MAP or
Laplace fit up to full NUTS sampling, and whole catalogs can be fitted at once.

JAXQSOFit is the spectrum-focused interface to the spectral model in
`jaxsedfit <https://jaxsedfit.readthedocs.io/>`_, which also fits spectra jointly
with broadband photometry.

.. note::

   JAXQSOFit is under active development; expect breaking API changes.

Citation
--------

If you use JAXQSOFit in published work, cite:

* PyQSOFit: Guo, Shen & Wang (2018), *PyQSOFit: Python code to fit the spectrum
  of quasars*, Astrophysics Source Code Library, ascl:1809.008
  (`ASCL <https://ascl.net/1809.008>`__), and Shen et al. (2019), ApJS, 241, 34
  (`ADS <https://ui.adsabs.harvard.edu/abs/2019ApJS..241...34S/abstract>`__);
* DSPS: Hearin et al. (2023), MNRAS, 521, 1741
  (`ADS <https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1741H/abstract>`__);
* dustmaps: Green (2018), JOSS, 3, 695
  (`ADS <https://ui.adsabs.harvard.edu/abs/2018JOSS....3..695G/abstract>`__).

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User guide

   model
   lines
   likelihood
   inference
   batch
   results
   pyqsofit
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Reference

   default_line_priors
   api/index
