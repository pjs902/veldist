import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'veldist'
copyright = '2025, Peter Smith'
author = 'Peter Smith'
version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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