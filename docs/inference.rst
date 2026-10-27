Inference methods
=================

``jaxqsofit`` offers three ways to fit one spectrum. Choose one with
``cfg.inference.method`` and call :meth:`jaxqsofit.JAXQSOFit.fit`. Every method
returns a :class:`jaxqsofit.FitResult`. ``result.samples`` maps each model
parameter to an array whose first axis indexes draws, so summaries,
``result.predict``, ``result.spectrum``, plots and ``save`` work the same way
whichever method produced them.

.. list-table::
   :header-rows: 1
   :widths: 18 22 20 40

   * - ``method``
     - Returns
     - Typical cost
     - Use it for
   * - ``"optax"``
     - the MAP (one point)
     - seconds, plus compilation
     - quick looks, smoke tests, starting points
   * - ``"nuts"``
     - MCMC draws
     - minutes
     - accurate posteriors
   * - ``"optax+nuts"`` (default)
     - MCMC draws, started at the MAP
     - minutes
     - accurate posteriors (recommended default)

A Laplace approximation (Gaussian draws at the MAP) is available for many spectra
at once through :func:`jaxqsofit.fit_batch`; see :doc:`batch`.

How to choose
-------------

* Start with ``"optax"`` to check that the model can describe the spectrum (look at
  the decomposition plot and the residuals) before spending time on uncertainties.
* Use ``"optax+nuts"`` for results you will publish, and whenever the posterior may
  be multimodal, strongly non-Gaussian or pushed against a prior bound. Broad-line
  decompositions often are: the split between broad and narrow components, Fe II
  and the host can trade off against each other.
* Use :func:`jaxqsofit.fit_batch` with ``method="laplace"`` when you need fast
  uncertainties for a catalog. Check them against ``"optax+nuts"`` on a few
  representative spectra before relying on them for a new kind of data.

MAP (``"optax"``)
-----------------

The maximum a posteriori point: the most probable parameters given the spectrum and
the priors, found with Adam (``map_steps``, ``learning_rate``). The optimization is
staged. The first stage, ``max(100, map_steps // 3)`` steps, fits only the power-law
continuum, reddening and host on pixels away from the strong emission lines. The
second stage fits the full model, lines included, from there.

.. code-block:: python

   cfg.inference.method = "optax"
   cfg.inference.map_steps = 1500
   cfg.inference.learning_rate = 1e-2
   result = JAXQSOFit(cfg).fit()
   result.samples["PL_slope"]  # shape (1,): the MAP

A MAP is a single point. It carries no uncertainty, and for skewed posteriors (for
example a weak line whose flux is consistent with zero) it can differ noticeably
from the posterior median.

NUTS (``"nuts"`` and ``"optax+nuts"``)
--------------------------------------

The No-U-Turn Sampler, a Hamiltonian Monte Carlo method, draws from the full
posterior with no Gaussian assumption. ``"optax+nuts"`` runs the MAP first
(``map_steps``) and starts the chains there, which shortens warmup; ``"nuts"``
starts from default initial values built from the priors and the line table.

.. code-block:: python

   cfg.inference.method = "optax+nuts"
   cfg.inference.map_steps = 800
   cfg.inference.num_warmup = 250
   cfg.inference.num_samples = 250
   cfg.inference.num_chains = 1
   cfg.inference.target_accept_prob = 0.82
   result = JAXQSOFit(cfg).fit()
   result.plot_trace()

The mass matrix is diagonal except for compact dense blocks for each emission-line
complex (``dense_mass=False``, ``line_block_dense_mass=True``, the defaults).
``max_tree_depth`` and ``warmup_max_tree_depth`` limit the trajectory length; see
:doc:`quickstart` for tuning a difficult tied-line spectrum.

Laplace approximation (batched fits only)
-----------------------------------------

``fit_batch(..., method="laplace")`` finds the MAP as above, then draws from a
Gaussian centered on it whose covariance is the inverse Hessian (the curvature) of
the negative log posterior. The Gaussian lives in NumPyro's unconstrained parameter
space, so draws of bounded parameters stay inside their priors. ``num_samples``
sets the number of draws. The Hessian costs little next to the optimization.

It is accurate when the posterior is close to Gaussian around its peak. It is
unreliable when:

* the posterior has several modes (the Gaussian only describes the one the MAP found);
* a parameter sits against a prior bound (for example a line amplitude at zero);
* parameters are tied together by a strongly curved degeneracy, such as the
  broad/narrow split of a line.

Each fitted spectrum records ``q.batch_diagnostics["hessian_min_eigenvalue"]``. If it
is zero or negative, the Hessian was not positive definite: the flat or non-convex
directions were given a very wide Gaussian, so treat that spectrum's uncertainties
as unreliable. Laplace intervals also reflect only the statistical noise of the
spectrum, not the systematic errors of the model (templates, line profiles,
host-AGN degeneracies), which often dominate.
