# Sphinx configuration of the Kosmulator documentation (https://kosmulator.readthedocs.io).
# The pages are Markdown (MyST); build locally with:
#     pip install -r docs/requirements.txt
#     sphinx-build -b html docs docs/_build/html
project = "Kosmulator"
author = "Renier T. Hough"
copyright = "2026, Renier T. Hough"

extensions = [
    "myst_parser",       # Markdown pages
    "sphinx_design",     # tabs and fold-out (dropdown) boxes
]

myst_enable_extensions = [
    "colon_fence",       # ::: blocks for notes, tabs and fold-outs
    "dollarmath",        # $...$ and $$...$$ maths
    "amsmath",
    "attrs_inline",
]
myst_heading_anchors = 3
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"   # links to sections, e.g. running.md#a-first-run

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "requirements.txt"]

html_theme = "sphinx_rtd_theme"
html_logo = "_static/kosmulator_logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Kosmulator documentation"
html_theme_options = {
    "logo_only": True,
    "style_nav_header_background": "#0b2233",   # the navy of the logo
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_context = {                                 # "Edit on GitHub" links
    "display_github": True,
    "github_user": "renierht",
    "github_repo": "Kosmulator",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
