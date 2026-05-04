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
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_favicon",
    "sphinx_togglebutton",
    "sphinx_contributors"
]

def linkcode_resolve(domain, info):
    """Return a URL to the source code on GitHub for the given object."""
    if domain != "py":
        return None

    import importlib
    import inspect

    module_name = info["module"]
    fullname = info["fullname"]

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None

    obj = mod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        source_file = inspect.getsourcefile(obj)
    except TypeError:
        return None

    if source_file is None:
        return None

    try:
        source_lines = inspect.getsourcelines(obj)
        lineno = source_lines[1]
    except (TypeError, OSError):
        lineno = None

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        rel_path = os.path.relpath(source_file, repo_root)
    except ValueError:
        return None

    url = f"https://github.com/calicarpa/krum/blob/main/{rel_path}"
    if lineno is not None:
        url += f"#L{lineno}"
    return url


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}


# Use MathJax to render math in HTML
mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_title = "Krum, the Library"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
]
html_js_files = [
    "external-links.js",
]
html_show_sourcelink = True
html_use_index = True

# Custom theme options
html_theme_options = {
    "page_layout": "default",
    "github_url": "https://github.com/calicarpa/krum",
    "discussion_url": "https://github.com/calicarpa/krum/discussions",
    "accent_color": "blue",
    "announcement": "Welcome to the new Krum documentation! Start with the <a href='/tutorials/index.html'>Tutorials</a> or dive into the <a href='/how-to/index.html'>How-to guides</a>.",
    "globaltoc_expand_depth": 2,
    "toctree_collapse": False,
    "toctree_includehidden": True,
    "nav_links_align": "center",
    "nav_links": [
        {
            "title": "Tutorials",
            "url": "tutorials/index",
        },
        {
            "title": "How-to guides",
            "url": "how-to/index",
            "children": [
                {
                    "title": "How to add an aggregator",
                    "url": "how-to/add-aggregator",
                },
                {
                    "title": "How to add an attack",
                    "url": "how-to/add-attack",
                },
                {
                    "title": "How to add a model",
                    "url": "how-to/add-model",
                },
                {
                    "title": "How to add a dataset",
                    "url": "how-to/add-dataset",
                },
                {
                    "title": "How to add a custom model",
                    "url": "how-to/add-custom-model",
                },
                {
                    "title": "How to add a custom dataset",
                    "url": "how-to/add-custom-dataset",
                }
            ],
        },
        {
            "title": "Explanation",
            "children": [
                {
                    "title": "Key concepts",
                    "url": "explanation/key-concepts",
                    "summary": "Understand the key concepts",
                },
                {
                    "title": "Debug mode",
                    "url": "explanation/debug-mode",
                    "summary": "Understand debug mode",
                },
                {
                    "title": "Native compilation",
                    "url": "explanation/native-compilation",
                    "summary": "Understand native compilation",
                },
                {
                    "title": "Tensor lifecycle",
                    "url": "explanation/tensor-lifecycle",
                    "summary": "Understand tensor lifecycle",
                },
                {
                    "title": "CLI format",
                    "url": "explanation/cli-format",
                    "summary": "Understand CLI format",
                },
            ],
        },
        {
            "title": "Reference",
            "children": [
                {
                    "title": "Architecture",
                    "url": "reference/architecture",
                    "summary": "Understanding the architecture of Krum",
                },
                {
                    "title": "Aggregators",
                    "url": "reference/aggregators/index",
                    "summary": "What are aggregators",
                },
                {
                    "title": "Attacks",
                    "url": "reference/attacks/index",
                    "summary": "What are attacks",
                },
                {
                    "title": "Experiments",
                    "url": "reference/experiments/index",
                    "summary": "What define an experiment",
                },
                {
                    "title": "Tools",
                    "url": "reference/tools/index",
                    "summary": "Which tools are available",
                },
                {
                    "title": "Native",
                    "url": "reference/native",
                    "summary": "What is native",
                },
            ],
        },
        {
            "title": "Contributors",
            "url": "contributors",
        },
    ],
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
