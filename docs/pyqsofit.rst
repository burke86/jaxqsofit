Comparison with PyQSOFit
========================

``jaxqsofit`` follows the decomposition of PyQSOFit: a smooth AGN continuum,
optional Fe II and Balmer-continuum components, a host-galaxy continuum, and
Gaussian emission-line complexes. The line table keeps PyQSOFit's concepts of line
and component names, velocity and width ties, and fixed flux ratios (see
:doc:`lines`).

The main differences:

**One joint model.** The continuum, host, Fe II, Balmer continuum and lines are fit
together in one probabilistic model, rather than fitting the continuum in fixed
rest-frame windows, subtracting it, and then fitting the lines. That avoids the
redshift-dependent systematics a staged workflow can introduce, reduces host-AGN
degeneracies, and gives more stable results for spectra with strong blending,
absorption or host contamination.

**Posterior uncertainties.** The model is built with JAX and NumPyro, so it can be
optimized or sampled with gradient-based inference (see :doc:`inference`). Posterior
draws are available for every component and line, so uncertainties propagate into
derived quantities such as broad-line FWHMs and luminosities.

**Host model.** PyQSOFit's host decomposition is closest to
``sfh_model="flexible"``, a free mixture of stellar templates. The default,
``sfh_model="delayed"``, imposes a physical delayed-:math:`\tau` star-formation
history instead, which gives more interpretable parameters at the cost of a stronger
assumption (see :doc:`model`).

**Instrumental resolution.** As in PyQSOFit, the instrumental-resolution correction
is optional. Here it is set by ``apply_instrumental_resolution`` and needs a
``resolving_power``; it is off by default, so enable it when you need intrinsic line
widths.

**PSF photometry.** Optional PSF magnitudes constrain the flux calibration and the
split between compact and extended light (see :doc:`likelihood`). This does not make
``jaxqsofit`` an SED fitter; ``jaxsedfit`` fits broadband photometry and spectra
together.
