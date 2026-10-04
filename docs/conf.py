"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Make krum submodules available as top-level imports for autodoc compatibility
for _mod, _pkg in (
    ("primitives", "krum.primitives"),
    ("aggregators", "krum.primitives.aggregators"),
    ("attacks", "krum.primitives.attacks"),
    ("models", "krum.primitives.models"),
    ("simulations", "krum.simulations"),
):
    sys.modules[_mod] = importlib.import_module(_pkg)

project = "Krum, the Library"
copyright = "2026"
author = "Arthur DANJOU, Mohammed Ammar SAID, El-Mahdi EL-MHAMDI, Sébastien ROUAULT, Peva BLANCHARD"


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
    "sphinx_contributors",
]


def linkcode_resolve(domain, info):
    """Resolve a documented object to a permalink to its source on GitHub.

    Used by the ``sphinx.ext.linkcode`` extension to turn ``[source]`` links
    in the rendered HTML into deep links to the corresponding file and line
    on the project's GitHub ``main`` branch.

    Args:
        domain: Sphinx object domain. Only ``"py"`` is handled; any other
            domain returns ``None``.
        info: Dictionary provided by Sphinx with ``"module"`` (the fully
            qualified module name) and ``"fullname"`` (the dotted
            attribute path inside the module).

    Returns:
        A ``github.com/calicarpa/krum/blob/main/<path>`` URL optionally
        anchored to a line (``#L<n>``), or ``None`` if the object cannot
        be resolved to a source file inside the repository.
    """
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
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# Annotations are strings everywhere (`from __future__ import annotations`), so
# autodoc renders the name as written. These map the names that cannot be
# resolved from the module they appear in: the project's own type aliases, which
# are referenced from several modules, and standard library classes whose
# runtime module is a private implementation detail (`pathlib._local.Path`,
# `_blake2.blake2b`) or differs from where they are documented.
autodoc_type_aliases = {
    "Blake2b": "hashlib.blake2b",
    "Path": "pathlib.Path",
    "ProcessPoolExecutor": "concurrent.futures.ProcessPoolExecutor",
}

# Targets a nitpicky build (`sphinx-build -n`) cannot resolve because nothing
# documents them. Each group has a reason; none is a broken cross-reference
# that qualifying the name would fix.
nitpick_ignore = [
    # The project's own type aliases. A PEP 695 `type X = Y` has no `py:class`
    # target in the Python domain, and `autodoc_type_aliases` substitutes a
    # name only when the annotation is exactly that name, so it cannot reach
    # the ones inside composites such as `Hash | str` or `PathLike | None`.
    ("py:class", "Hash"),
    ("py:class", "PathLike"),
    ("py:class", "RunCallable"),
    ("py:class", "Runner"),
    ("py:class", "krum.simulations.decentralised.StepResultT"),
    ("py:obj", "krum.simulations.decentralised.StepResultT"),
    # Standard library classes reached the same way: autodoc renders the name
    # as written, and these appear only inside composite annotations.
    ("py:class", "Blake2b"),
    ("py:class", "Path"),
    # Members that exist but carry no documentation of their own: instance
    # attributes declared without a docstring, and a method only some
    # subclasses provide. Documenting them is what would retire these.
    ("py:attr", "byzantine_reach"),
    ("py:attr", "f"),
    ("py:attr", "krum.simulations.decentralised.monna_icml_2023.MonnaSimulation.byzantine_reach"),
    ("py:attr", "model"),
    ("py:attr", "parameters"),
    ("py:attr", "step_index"),
    ("py:attr", "test_loader"),
    ("py:meth", "copy_parameters_to_model"),
]


# Use MathJax v3 to render math in HTML
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

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
    "announcement": "Welcome to the new Krum documentation!",
    "globaltoc_expand_depth": 2,
    "toctree_collapse": False,
    "toctree_includehidden": True,
    "nav_links_align": "center",
    "nav_links": [
        {
            "title": "Quickstart",
            "url": "quickstart",
        },
        {
            "title": "Tutorials",
            "url": "tutorials/index",
        },
        {
            "title": "Reference",
            "children": [
                {
                    "title": "Orchestration",
                    "url": "reference/orchestration/index",
                    "summary": "Running and managing simulations and metrics",
                },
                {
                    "title": "Primitives",
                    "url": "reference/primitives/index",
                    "summary": "Core abstractions",
                    "children": [
                        {
                            "title": "Models",
                            "url": "reference/primitives/models/index",
                            "summary": "Standard models for simulations",
                        },
                        {
                            "title": "Aggregators",
                            "url": "reference/primitives/aggregators/index",
                            "summary": "Byzantine-resilient gradient aggregation rules",
                        },
                        {
                            "title": "Attacks",
                            "url": "reference/primitives/attacks/index",
                            "summary": "Byzantine attack strategies for evaluation",
                        },
                    ],
                },
                {
                    "title": "Simulations",
                    "url": "reference/simulations/index",
                    "summary": "Reproducing published experiments in centralised or decentralised settings",
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

mathjax3_config = {
    "tex": {
        "macros": {
            "argmin": r"\mathop{\mathrm{arg\,min}}",
        }
    }
}
