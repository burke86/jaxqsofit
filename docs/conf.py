from __future__ import annotations

import os
import sys
from datetime import datetime

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

ROOT = os.path.abspath('..')
SRC = os.path.join(ROOT, 'src')
DOCS_EXT = os.path.abspath('_ext')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if DOCS_EXT not in sys.path:
    sys.path.insert(0, DOCS_EXT)

project = 'JAXQSOFit'
author = 'JAXQSOFit contributors'
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'nbsphinx_link',
    'default_line_table',
]

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_class_signature = 'mixed'
autodoc_typehints = 'description'
autodoc_type_aliases = {
    'FitConfig': 'jaxqsofit.config.FitConfig',
}
autodoc_preserve_defaults = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep docs build light and robust on RTD.
autodoc_mock_imports = [
    'jax',
    'jaxlib',
    'numpyro',
    'optax',
    'dsps',
    'diffmah',
    'diffstar',
    'dustmaps',
    'extinction',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['default-line-table.css']
html_title = 'JAXQSOFit Documentation'
html_show_sourcelink = False
html_theme_options = {
    'path_to_docs': 'docs',
    'repository_url': 'https://github.com/burke86/jaxqsofit',
    'repository_branch': 'main',
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_download_button': True,
}

# Render saved tutorial outputs without rerunning expensive inference during
# documentation builds. Set NBSPHINX_EXECUTE=always explicitly when a fresh
# local notebook execution is desired.
nbsphinx_execute = os.environ.get('NBSPHINX_EXECUTE', 'never')
nbsphinx_allow_errors = False
