# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from sphinx.ext.apidoc import main
import pathlib

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../..'))

# -- Docs setup --------------------------------------------------------------
def run_apidoc(_):
    package_dir = pathlib.Path(__file__).parent / "../../meze"
    main(["-o", str(pathlib.Path(__file__).parent), str(package_dir), "--force"])

def setup(app):
    app.connect("builder-inited", run_apidoc)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'meze'
copyright = '2025, J. Jasmin Guven'
author = 'J. Jasmin Guven'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',          # pull docstrings from your code
    'sphinx.ext.napoleon',         # support Google/NumPy-style docstrings
    'sphinx_autodoc_typehints',    # include type hints in documentation
    'sphinx.ext.viewcode',         # add links to highlighted source code
]

autodoc_mock_imports = ["MDAnalysis", "BioSimSpace", "bss"]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,      
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
