import importlib.util
from pathlib import Path

import numpy as np

from jaxsedfit.model import _cigale_delayed_sfh_shape, _delayed_sfh_cumulative_mass

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "chimera_spectra_validation.py"
_spec = importlib.util.spec_from_file_location("chimera_spectra_validation", _PATH)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def test_jname_to_deg():
    ra, dec = validation._jname_to_deg("J013549.53+241149.7")
    assert np.isclose(ra, 15.0 * (1 + 35 / 60 + 49.53 / 3600))
    assert np.isclose(dec, 24 + 11 / 60 + 49.7 / 3600)
    ra, dec = validation._jname_to_deg("J021558.40-011135.9")
    assert np.isclose(dec, -(1 + 11 / 60 + 35.9 / 3600))
    assert validation._sdss_name("013549.53+241149.7_383517_0.0001") == "J013549.53+241149.7"


def test_log_sfr_matches_jaxsedfit_delayed_sfh():
    age, tau, formed = 5.0, 2.0, 3.0e10
    pred = {"sfh_age_gyr": np.array([age]), "sfh_tau_gyr": np.array([tau]), "formed_stellar_mass": np.array([formed])}
    expected = formed * float(_cigale_delayed_sfh_shape(age, tau, age)) / (float(_delayed_sfh_cumulative_mass(age, tau)) * 1.0e9)
    assert np.isclose(validation._log_sfr(pred), np.log10(expected))


def test_log_lbol_uses_shen11_corrections():
    pred = {f"log_lambda_Llambda_{w}_agn": np.array([44.0 + i]) for i, w in enumerate((5100, 3000, 1350))}
    assert np.isclose(validation._log_lbol(pred, 0.5), 44.0 + np.log10(9.26))
    assert np.isclose(validation._log_lbol(pred, 1.0), 45.0 + np.log10(5.15))
    assert np.isclose(validation._log_lbol(pred, 2.5), 46.0 + np.log10(3.81))
