# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Polnet'
copyright = '2026, Juan Diego Gallego Nicolas'
author = 'Juan Diego Gallego Nicolas'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Pulls in documentation from docstrings
    'sphinx.ext.napoleon',          # Parses Google-style docstrings
    'sphinx.ext.viewcode',          # Adds links to highlighted source code
    'sphinx.ext.mathjax',           # Renders LaTeX math in docstrings
    'sphinx_autodoc_typehints',     # Renders type hints in signatures
    'sphinx_rtd_theme',             # (Optional) Uses the ReadTheDocs theme
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
