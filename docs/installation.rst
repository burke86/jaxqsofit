Installation
============

Requirements
------------

- Python 3.10+
- JAX/JAXLIB
- NumPyro
- DSPS
- dustmaps

Setup notes
-----------

1. Configure dustmaps SFD (one-time):

.. code-block:: bash

   python setup.py fetch --map-name=sfd

Then ensure dustmaps points to the directory containing the downloaded SFD maps.

2. Set ``dsps_ssp_fn`` to your DSPS HDF5 SSP template file path when fitting
   (for example ``tempdata.h5``).

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/burke86/jaxqsofit.git
   cd jaxqsofit
   pip install -e .

Optional test dependencies
--------------------------

.. code-block:: bash

   pip install pytest pytest-cov astroquery
