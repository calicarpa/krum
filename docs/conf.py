# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Krum, the Library"
copyright = "2026"
author = "Peva BLANCHARD, Arthur DANJOU, El-Mahdi EL-MHAMDI, Sébastien ROUAULT, Mohammed Ammar SAID"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_favicon",
    "sphinx_togglebutton",
]


# Use MathJax to render math in HTML
mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"  # "pydata_sphinx_theme"
html_title = "Krum, the Library"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
]
html_show_sourcelink = False
html_use_index = True

html_theme_options = {
    "header_links_before_dropdown": 5,
    "navigation_depth": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/calicarpa/krum",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
    "search_bar_text": "Search docs...",
    "navbar_persistent": ["search-button-field"],
}

# html_favicon = "_static/favicon.ico"

napoleon_custom_sections = [
    ("Initialization parameters", "params_style"),
    ("Input parameters", "params_style"),
    ("Calling the instance", "rubric_style"),
    ("Returns", "params_style"),
]

latex_elements = {
    "preamble": r"""
        \usepackage{amsmath}
        \newcommand{\argmin}{\mathop{\mathrm{arg\,min}}}
    """
}

mathjax_config = {
    "TeX": {
        "Macros": {
            "argmin": r"\mathop{\mathrm{arg\,min}}",
        }
    }
}
