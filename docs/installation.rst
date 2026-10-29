Installation
============

``jaxqsofit`` requires Python 3.10 or newer. Install it from source:

.. code-block:: bash

   git clone https://github.com/burke86/jaxqsofit.git
   cd jaxqsofit
   pip install -e .

This installs the dependencies, including JAX, NumPyro, DSPS, ``dustmaps`` and the
pinned version of ``jaxsedfit`` that provides the spectral model. The default JAX
install runs on CPU; for a GPU, install ``jax`` following the official JAX
instructions for your CUDA setup.

For the tests and the tutorials:

.. code-block:: bash

   pip install pytest pytest-cov astroquery

SSP templates
-------------

The host-galaxy model needs a DSPS SSP template file, passed as
``HostConfig(dsps_ssp_fn=...)``. A continuum-only template is recommended, because
emission lines are modeled separately:

.. code-block:: bash

   curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5

Dust maps
---------

Milky Way dereddening is on by default (``Observation.apply_mw_deredden``) and uses
the SFD maps through ``dustmaps``, with the target's ``ra`` and ``dec``. Download
the maps once:

.. code-block:: python

   from dustmaps.config import config
   config["data_dir"] = "/path/to/dustmaps"

   import dustmaps.sfd
   dustmaps.sfd.fetch()
