import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'veldist'
copyright = '2025-2026, Peter Smith'
author = 'Peter Smith'

# Single source of truth is pyproject.toml — do not hardcode.
from importlib.metadata import version as _v
release = _v('veldist')
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_copybutton',
]

templates_path = ['_templates']
# 'superpowers' holds working notes (plans, specs, decision records), not
# published documentation. It is gitignored, but Sphinx still finds it on a
# local build and emits a toctree warning per file, which buries real warnings.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'superpowers']

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/pjs902/veldist",
    "use_repository_button": True,
}

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]