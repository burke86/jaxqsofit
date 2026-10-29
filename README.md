# JAXQSOFit

<p align="center">
  <img src="logo.png" alt="JAXQSOFit logo" width="220">
</p>

JAXQSOFit is a Bayesian quasar spectral fitting package built with JAX and NumPyro. It is a JAX port of the PyQSOFit decomposition, fitted jointly in one probabilistic model rather than in stages:

- a power-law AGN continuum with intrinsic reddening
- host-galaxy decomposition from DSPS SSP templates, with stellar kinematics
- UV/optical Fe II templates and the Balmer continuum
- tied Gaussian emission lines with PyQSOFit-style line tables
- built-in BAL absorption for N V, Si IV and C IV
- a Student-t likelihood with fitted error jitter, and a multiplicative polynomial for flux-calibration errors
- user-defined custom components, such as an alternative iron template or a non-Gaussian line profile
- optional PSF magnitudes to constrain the flux calibration and the host fraction

Fits return posterior draws of every component and line measurement, from a fast MAP or Laplace fit up to full NUTS sampling, and `fit_batch` fits whole catalogs with one compiled program.

JAXQSOFit is the spectrum-focused interface to the spectral model in [jaxsedfit](https://github.com/burke86/jaxsedfit), which also fits spectra jointly with broadband photometry. The `jaxqsofit.model`, `jaxqsofit.components`, `jaxqsofit.defaults` and related modules remain importable as aliases of their `jaxsedfit` implementations.

**Documentation: [https://jaxqsofit.readthedocs.io/](https://jaxqsofit.readthedocs.io/)**

> `JAXQSOFit` is under active development; expect breaking API changes.

## Install

`jaxqsofit` requires Python 3.10 or newer:

```bash
git clone https://github.com/burke86/jaxqsofit.git
cd jaxqsofit
pip install -e .
curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5
```

The second command downloads a continuum-only DSPS SSP template for the host model, passed as `HostConfig(dsps_ssp_fn="tempdata.h5")`. Milky Way dereddening, on by default, also needs the SFD dust maps; see the [installation page](https://jaxqsofit.readthedocs.io/en/latest/installation.html).

## Example

```python
import numpy as np
import jaxqsofit
from astroquery.sdss import SDSS
from astropy import units as u
from astropy.coordinates import SkyCoord

# Fetch the SDSS spectrum of NGC 5548
coord = SkyCoord.from_name("NGC 5548")
xid = SDSS.query_region(coord, spectro=True, radius=5 * u.arcsec)
sp = SDSS.get_spectra(matches=xid)[0]
tb = sp[1].data

cfg = jaxqsofit.FitConfig(
    observation=jaxqsofit.Observation(
        object_id="ngc5548",
        redshift=float(sp[2].data["z"][0]),
        ra=coord.ra.deg,                       # for Milky Way dereddening
        dec=coord.dec.deg,
    ),
    spectroscopy=jaxqsofit.SpectroscopyData(
        wave_obs=10 ** tb["loglam"],           # observed-frame wavelength [A]
        fluxes=tb["flux"],                     # f_lambda [1e-17 erg/s/cm2/A]
        errors=1.0 / np.sqrt(tb["ivar"]),
    ),
    host=jaxqsofit.HostConfig(enabled=True, dsps_ssp_fn="tempdata.h5"),
    inference=jaxqsofit.InferenceConfig(method="optax+nuts"),
)

q = jaxqsofit.JAXQSOFit(cfg)
result = q.fit()
result.spectrum.line_groups["Hb_br"].total_flux_erg_s_cm2  # posterior draws
```

The [quickstart](https://jaxqsofit.readthedocs.io/en/latest/quickstart.html) continues from here. To start with a notebook, see the [first tutorial](notebooks/01_jaxqsofit_tutorial.ipynb); all tutorials are listed on the [tutorials page](https://jaxqsofit.readthedocs.io/en/latest/tutorials.html).

## Citation

If you use JAXQSOFit in published work, please cite:

- PyQSOFit: Guo, Shen & Wang (2018), [ascl:1809.008](https://ascl.net/1809.008), and Shen et al. (2019), *ApJS*, 241, 34 ([ADS](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...34S/abstract))
- DSPS: Hearin et al. (2023), *MNRAS*, 521, 1741 ([ADS](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1741H/abstract))
- dustmaps: Green (2018), *JOSS*, 3, 695 ([ADS](https://ui.adsabs.harvard.edu/abs/2018JOSS....3..695G/abstract))
