Quickstart
==========

Minimal fitting example
-----------------------

.. code-block:: python

   import numpy as np
   from jaxqsofit import QSOFit

   # Example arrays
   lam = np.linspace(3800.0, 9200.0, 2000)
   flux = 50.0 + 0.002 * (lam - 6000.0)
   err = np.full_like(flux, 0.5)
   z = 0.1

   q = QSOFit(lam=lam, flux=flux, err=err, z=z)
   q.Fit(
       fit_method='nuts',
       fit_lines=True,
       decompose_host=True,
       fit_fe=True,
       fit_bc=True,
       fit_poly=False,
       save_result=False,
       plot_fig=True,
   )

Fast mode
---------

For a fast MAP-style fit, use:

.. code-block:: python

   q.Fit(
       fit_method='optax',
       optax_steps=1500,
       optax_lr=1e-2,
       save_result=False,
       plot_fig=True,
   )

Hybrid mode
-----------

Warm-start with Optax, then run NUTS:

.. code-block:: python

   q.Fit(
       fit_method='optax+nuts',
       optax_steps=800,
       nuts_warmup=200,
       nuts_samples=400,
   )
