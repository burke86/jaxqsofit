Inference methods
=================

``jaxqsofit`` offers four ways to fit one spectrum. Choose one with
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
   * - ``"laplace"``
     - Gaussian draws at the MAP
     - MAP + one Hessian
     - fast uncertainties for well-constrained, near-Gaussian posteriors; catalogs
   * - ``"nuts"``
     - MCMC draws
     - minutes
     - accurate posteriors
   * - ``"optax+nuts"`` (default)
     - MCMC draws, started at the MAP
     - minutes
     - accurate posteriors (recommended default)

To fit many spectra at once with ``"optax"`` or ``"laplace"``, see :doc:`batch`.

How to choose
-------------

* Start with ``"optax"`` to check that the model can describe the spectrum (look at
  the decomposition plot and the residuals) before spending time on uncertainties.
* Use ``"optax+nuts"`` for results you will publish, and whenever the posterior may
  be multimodal, strongly non-Gaussian or pushed against a prior bound. Broad-line
  decompositions often are: the split between broad and narrow components, Fe II
  and the host can trade off against each other.
* Use ``"laplace"`` when you need fast uncertainties, for example for a catalog
  with :func:`jaxqsofit.fit_batch`. Check them against ``"optax+nuts"`` on a few
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
complex and its shared ordered-width hierarchy (``dense_mass=False``,
``line_block_dense_mass=True``, the defaults); the continuum, host and unrelated line
coordinates stay diagonal. ``dense_mass=True`` requests one fully dense matrix
instead, which usually needs much more warmup; ``line_block_dense_mass=False`` makes
it fully diagonal.

``max_tree_depth`` limits the trajectory length during both warmup and sampling.
For a difficult tied-line spectrum, try a longer but shallower warmup before raising
the limit for the retained draws:

.. code-block:: python

   cfg.inference.num_warmup = 500
   cfg.inference.warmup_max_tree_depth = 7
   cfg.inference.max_tree_depth = 8

The extra warmup draws matter: with only 250 warmup draws, a depth-7 limit can leave
the line-block covariance under-adapted.

Laplace approximation (``"laplace"``)
-------------------------------------

The MAP as above, followed by a Gaussian centered on it whose covariance is the
inverse Hessian (the curvature) of the negative log posterior. The Gaussian lives
in NumPyro's unconstrained parameter space, so draws of bounded parameters stay
inside their priors. ``num_samples`` sets the number of draws. The Hessian costs
little next to the optimization.

.. code-block:: python

   cfg.inference.method = "laplace"
   cfg.inference.map_steps = 2000
   cfg.inference.num_samples = 400
   q = JAXQSOFit(cfg)
   result = q.fit()
   q.approx_result["hessian_min_eigenvalue"]

:func:`jaxqsofit.fit_batch` with ``method="laplace"`` does the same for many
spectra at once.

It is accurate when the posterior is close to Gaussian around its peak. It is
unreliable when:

* the posterior has several modes (the Gaussian only describes the one the MAP found);
* a parameter sits against a prior bound (for example a line amplitude at zero);
* parameters are tied together by a strongly curved degeneracy, such as the
  broad/narrow split of a line.

Each fit records ``q.approx_result["hessian_min_eigenvalue"]``. Below one, some
directions were flat or non-convex (or the MAP had not converged): their curvature
is raised to one, a Gaussian of roughly the prior's width in unconstrained
coordinates, so treat uncertainties along them as unreliable. Zero or negative means
the Hessian was not positive definite at all. Laplace intervals also reflect only the statistical noise of the
spectrum, not the systematic errors of the model (templates, line profiles,
host-AGN degeneracies), which often dominate.
