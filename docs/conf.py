# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Replace "MyProject" / "precisionai.myproject" with your actual project name before shipping.

import os
import shutil
import subprocess
import sys

# Add project root so 'precisionai' can be imported for autodoc.
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

# Copy root README.md as first chapter of the docs/PDF.
_docs_dir = os.path.dirname(os.path.abspath(__file__))
_readme_src = os.path.join(_root, "README.md")
_readme_copy = os.path.join(_docs_dir, "readme.md")
if os.path.isfile(_readme_src):
    shutil.copy2(_readme_src, _readme_copy)


def _get_version_from_git() -> str:
    """Use current git tag as version (e.g. 0.1.0); 0.0.0 if not on a tag."""
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            cwd=_root,
            timeout=5,
            check=False,
        )
        if out.returncode != 0 or not out.stdout:
            return "0.0.0"
        tag = out.stdout.strip()
        return tag.lstrip("v") if tag.startswith("v") else tag
    except Exception:
        return "0.0.0"


# -- Version from current git tag (0.0.0 if not on a tag) --------------------
version = release = _get_version_from_git()

print("Version: ", version)
print("Release: ", release)

# -- General configuration ----------------------------------------------------
project = "PAI MyProject"
copyright = "Precision AI"
author = "Precision AI"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Support Markdown files alongside RST.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Avoid "more than one target found" for re-exported names.
suppress_warnings = ["ref.python"]

# Napoleon settings for NumPy-style docstrings.
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- Options for HTML output --------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]

# -- Options for LaTeX/PDF output ---------------------------------------------
latex_documents = [
    (
        "index",
        "documentation.tex",
        "PAI MyProject",
        author,
        "manual",
        True,
    ),
]

# Logo for cover and header (optional — remove if no logo.png is present).
_logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
latex_additional_files = ["assets/logo.png"] if os.path.isfile(_logo_path) else []
_logo_latex = "logo" if latex_additional_files else ""

_header_right = (
    r"\raisebox{-0.2\height}{\includegraphics[height=0.45cm]{%s.png}}\quad PAI MyProject --- %s"
    % (_logo_latex, version)
    if _logo_latex
    else "PAI MyProject --- %s" % version
)

_latex_preamble = r"""
%% Allow deeply nested lists (must come early; avoids "Too deeply nested" from autodoc)
\usepackage{enumitem}
\setlistdepth{99}
\usepackage{graphicx}
\usepackage{fancyhdr}
%% Fix fancyhdr: headheight and topmargin so header fits
\setlength{\headheight}{14.323pt}
\addtolength{\topmargin}{-2.323pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[RE,LO]{\leftmark}
\fancyhead[LE,RO]{%s}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}
\fancypagestyle{normal}{
  \fancyhf{}
  \fancyhead[RE,LO]{\leftmark}
  \fancyhead[LE,RO]{%s}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0pt}
}
""" % (_header_right, _header_right)

if _logo_latex:
    _latex_maketitle = (
        r"""
\makeatletter
\begin{center}
\includegraphics[width=4cm]{%s.png}\par
\vspace{1.2cm}
{\LARGE\sffamily\bfseries \@title \par}
\vspace{0.5cm}
{\Large \@author \par}
\vspace{0.8cm}
\releasename: \version \par
\@date
\end{center}
\makeatother
"""
        % _logo_latex
    )
else:
    _latex_maketitle = r"""
\makeatletter
\begin{center}
{\LARGE\sffamily\bfseries \@title \par}
\vspace{0.5cm}
{\Large \@author \par}
\vspace{0.8cm}
\releasename: \version \par
\@date
\end{center}
\makeatother
"""

_latex_fontpkg = r"\usepackage{mathptmx}\usepackage{helvet}\renewcommand{\ttdefault}{pcr}"

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "fontpkg": _latex_fontpkg,
    "releasename": "Version",
    "tableofcontents": r"\sphinxtableofcontents",
    "preamble": _latex_preamble,
    "maketitle": _latex_maketitle,
    "extraclassoptions": "openany",
}

latex_logging = False
