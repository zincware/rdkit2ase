# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import typing as t
from importlib.metadata import metadata

pkg_metadata = metadata("rdkit2ase")

project = "rdkit2ase"
author = "Fabian Zills"
version = pkg_metadata["Version"]
release = f"v{version}"
copyright = f"2025, {author}"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = project
html_short_title = project
# html_favicon = "_static/mlipx-favicon.svg"

html_theme_options: t.Dict[str, t.Any] = {
    # "light_logo": "mlipx-light.svg",
    # "dark_logo": "mlipx-dark.svg",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/zincware/rdkit2ase",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
    "source_repository": "https://github.com/zincware/rdkit2ase",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
}

# font-awesome logos
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]


# -- Options for sphinx_copybutton -------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/

copybutton_prompt_text = r">>> |\.\.\. |\(.*\) \$ "
copybutton_prompt_is_regexp = True
