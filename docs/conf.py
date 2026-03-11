from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.abspath('..')
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

project = 'JaxQSOFit'
author = 'JaxQSOFit contributors'
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep docs build light and robust on RTD.
autodoc_mock_imports = [
    'jax',
    'jaxlib',
    'numpyro',
    'optax',
    'dsps',
    'dustmaps',
    'extinction',
    'astropy',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'JaxQSOFit Documentation'
