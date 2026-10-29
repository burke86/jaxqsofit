Emission lines
==============

Emission lines are Gaussian components defined by a line table: one row per line,
with ties between lines in the PyQSOFit style. :doc:`default_line_priors` lists the
built-in tables with their priors.

Line lists
----------

The default table, :data:`jaxqsofit.defaults.DEFAULT_LINE_PRIOR_ROWS`, is the
broad-line AGN line list. Lines are active when their complex overlaps the fitted
rest-frame wavelength range. Two optional lists can be added with the line
configuration:

.. code-block:: python

   cfg.lines.include_elg_narrow_lines = True
   cfg.lines.include_high_ionization_lines = True

These switches apply whether the priors are generated automatically or supplied in
``cfg.prior_config``.

``include_elg_narrow_lines``
   Appends :data:`jaxqsofit.defaults.DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS`, a denser
   narrow-line list for emission-line-galaxy and host-dominated spectra: the
   resolved [O II] doublet, [Ne III], more narrow Balmer lines, He I, [O I], [N II],
   [S II], near-IR Paschen lines and [S III]. Leave it off for ordinary broad-line
   quasars when these weak lines are outside the wavelength range or would only add
   unconstrained amplitudes.

``include_high_ionization_lines``
   Appends :data:`jaxqsofit.defaults.DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS`: the
   high-ionization and coronal lines [Ne V], [Fe VII] and [Fe X]. Enable it when
   those lines are expected or visible.

``cfg.lines.use_broad_lines`` and ``cfg.lines.use_narrow_lines`` turn the broad or
narrow components off, and ``cfg.lines.enabled = False`` turns off the line model.

Editing the line table
----------------------

Copy the default table before editing it; it is a module-level list of mutable
dictionaries:

.. code-block:: python

   from copy import deepcopy

   from jaxqsofit import PriorConfig
   from jaxqsofit.defaults import DEFAULT_LINE_PRIOR_ROWS

   line_table = deepcopy(DEFAULT_LINE_PRIOR_ROWS)
   for row in line_table:
       if row["linename"] == "Hb_br":
           row["ngauss"] = 3

   if cfg.prior_config is None:
       cfg.prior_config = PriorConfig()
   cfg.prior_config.lines.table = line_table

The most commonly edited fields are:

``linename``
   The component name used in results and plots. Names containing ``"_br"`` are
   broad components; other built-in names are narrow.

``compname``
   The line complex, which scopes the tie indices: reusing an index in different
   complexes does not tie unrelated lines.

``ngauss``
   The number of Gaussian components made from the row. The default table already
   uses several broad components for ``Ha_br``, ``Hb_br``, ``MgII_br``, ``CIV_br``
   and ``Lya_br``. Components are named ``Hb_br_1``, ``Hb_br_2``, and so on.

``maxsca`` and ``maxsig``
   The upper bounds of the amplitude and width priors (widths are in
   :math:`\ln\lambda`).

Ties
----

The tie columns follow PyQSOFit's conventions, scoped by ``compname``:
``vindex`` ties velocity shifts, ``windex`` ties Gaussian widths and ``findex`` ties
amplitudes, i.e. flux ratios. Only positive indices create a shared group; zero
means the row is independent. Within a positive ``findex`` group, ``fvalue`` sets
each component's fixed relative peak amplitude. For rows with ``findex=0``,
``fvalue`` only sets the initial amplitude scale of the independent amplitude prior.

Narrow-line kinematics
----------------------

By default (``cfg.lines.pool_narrow_centroids = True``) the narrow lines are pooled
into three kinematic families: low-ionization (systemic), high-ionization and
coronal. All lines in a family share one centroid and one FWHM, across complexes.
The low-ionization velocity has a zero-centered normal prior with a 250 km/s scale;
the high-ionization velocity is offset from it with a 150 km/s scale, and the
coronal velocity from the high-ionization one with a 250 km/s scale. Broad
components, wings and outflow components are not pooled, and explicit ``vindex``
ties are kept.

Set ``pool_narrow_centroids=False`` to use the line table's own kinematics: untied
narrow groups then get independent centroids and widths, while explicit ``vindex``
and ``windex`` ties remain.
