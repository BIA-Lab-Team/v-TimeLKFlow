"""One-off generator: converts the replication scripts in examples/ into
Jupyter notebooks under examples/notebooks/, splitting on the scripts'
existing `# -- section --` comment banners. Not part of the installed
package or the test suite -- run manually whenever a source script changes
and the notebooks need regenerating.
"""

import ast
import os
import re

import nbformat as nbf

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "notebooks")

SECTION_RE = re.compile(r"^#\s*[─\-]{2,}\s*(.*?)\s*[─\-]{2,}\s*$")
DECORATION_RE = re.compile(r"^#\s*[─\-]{5,}\s*$")
PLAIN_TITLE_RE = re.compile(r"^#\s*(.+?)\s*$")

INTROS = {
    "fig2_fig3_replication.py": (
        "01_fig2_fig3_replication",
        "Synthetic Models 1 & 2 (Zhou et al. 2024, Figures 2-3)",
        "A 3-variable AR(1) system whose causal links ramp on, hold, and "
        "ramp back off. Model 1 (Fig. 2) includes a worked example of a "
        "*spurious* bivariate link that vanishes once conditioned on the "
        "confounder -- the core motivation for multivariate over pairwise "
        "causality analysis. Model 2 (Fig. 3) is a different 3-link network.",
    ),
    "fig4_model3_replication.py": (
        "02_fig4_model3_replication",
        "Synthetic Model 3 (Zhou et al. 2024, Figure 4)",
        "A 5-variable network that switches on and *stays* on (no "
        "ramp-down), testing multivariate conditioning with several "
        "converging/mediating paths at once.",
    ),
    "fig_appendixC_replication.py": (
        "03_fig_appendixC_replication",
        "Null-link tests (Zhou et al. 2024, Appendix C)",
        "For each of the three synthetic models, plots every pair with "
        "*no* preset causal link (17 pairs total) and confirms the "
        "estimated flow stays non-significant throughout -- i.e. no false "
        "positives.",
    ),
    "synthetic_replication.py": (
        "04_synthetic_replication",
        "Supplementary validation scenarios",
        "Additional scenarios not tied to a specific paper figure: an "
        "abrupt regime shift, a trivariate confounder, and a sinusoidally "
        "time-varying coupling strength.",
    ),
}

RUNTIME_NOTE = (
    "This notebook uses a reduced ensemble size (`N_REAL`) compared to the "
    "paper's 1000 realizations, purely for tractability -- expect roughly "
    "1-5 minutes to run top to bottom, and slightly noisier curves than the "
    "paper's own smoother averages."
)


def split_source(source: str) -> list[tuple[str, str]]:
    """Split a script's source into (title, code) chunks at section-comment
    banners. Handles two banner styles used across the example scripts:
    single-line `# -- Title --` banners, and three-line banners (a
    decoration-only line, a plain title comment, another decoration-only
    line). Returns a list where each entry is a chunk of code, with an
    empty title for the leading chunk before the first banner."""
    lines = source.splitlines()
    chunks = []
    current_title = ""
    current_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        m = SECTION_RE.match(line)
        if m:
            if current_lines:
                chunks.append((current_title, "\n".join(current_lines).strip("\n")))
            current_title = m.group(1)
            current_lines = []
            i += 1
            continue
        if (DECORATION_RE.match(line) and i + 2 < n
                and PLAIN_TITLE_RE.match(lines[i + 1].strip())
                and not SECTION_RE.match(lines[i + 1].strip())
                and DECORATION_RE.match(lines[i + 2].strip())):
            if current_lines:
                chunks.append((current_title, "\n".join(current_lines).strip("\n")))
            current_title = PLAIN_TITLE_RE.match(lines[i + 1].strip()).group(1)
            current_lines = []
            i += 3
            continue
        current_lines.append(lines[i])
        i += 1
    if current_lines:
        chunks.append((current_title, "\n".join(current_lines).strip("\n")))
    return chunks


def _adapt_for_notebook(source: str) -> str:
    """Adjust script-specific idioms that don't work (or don't make sense)
    in a notebook: `__file__` isn't defined in Jupyter, the "Agg" backend
    disables inline figure display, and `plt.close(fig)` right after
    `savefig` would close the figure before Jupyter's inline backend
    displays it at the end of the cell."""
    # No sys.path hack needed once mtvlk is pip-installed (which the
    # tutorial README instructs before opening the notebooks).
    source = re.sub(
        r"^sys\.path\.insert\(0, os\.path\.join\(os\.path\.dirname\(__file__\), \"\.\.\"\)\)\n",
        "", source, flags=re.MULTILINE,
    )
    # fig_appendixC_replication.py cross-imports its sibling example
    # scripts via sys.path.insert(0, os.path.dirname(__file__)) -- __file__
    # doesn't exist in a notebook, and the notebook's cwd could be the repo
    # root, examples/, or examples/notebooks/ depending on how Jupyter was
    # launched, so search a few candidate locations instead of assuming one.
    source = source.replace(
        'sys.path.insert(0, os.path.dirname(__file__))\n',
        '_candidates = [os.getcwd(), os.path.join(os.getcwd(), "examples"),\n'
        '               os.path.dirname(os.getcwd()),\n'
        '               os.path.join(os.path.dirname(os.getcwd()), "examples")]\n'
        'for _c in _candidates:\n'
        '    if os.path.exists(os.path.join(_c, "fig2_fig3_replication.py")):\n'
        '        sys.path.insert(0, _c)\n'
        '        break\n'
        'else:\n'
        '    raise FileNotFoundError(\n'
        '        "Could not locate examples/ (containing fig2_fig3_replication.py). "\n'
        '        "Run this notebook with Jupyter\'s working directory set to the repo "\n'
        '        "root or the examples/ directory."\n'
        '    )\n',
    )
    # Use the notebook's own inline backend instead of forcing Agg.
    source = source.replace('matplotlib.use("Agg")\n', "")
    # Let figures stay open so Jupyter's inline backend displays them.
    source = re.sub(r"^\s*plt\.close\(\w+\)\n", "", source, flags=re.MULTILINE)
    # Any remaining __file__ use (e.g. "save output next to this script")
    # becomes "save to the current working directory" in a notebook.
    source = source.replace("os.path.dirname(__file__)", "os.getcwd()")
    return source


def build_notebook(py_path: str, out_path: str, nb_title: str, subtitle: str, desc: str) -> None:
    with open(py_path, "r", encoding="utf-8-sig") as f:
        source = f.read()

    module = ast.parse(source)
    docstring = ast.get_docstring(module) or ""

    # Drop the module docstring from the source before splitting into cells
    # (it's rendered as markdown instead).
    if docstring:
        # Remove the first string-literal expression statement.
        first = module.body[0]
        source_lines = source.splitlines()
        source = "\n".join(source_lines[first.end_lineno:]).lstrip("\n")

    source = _adapt_for_notebook(source)

    nb = nbf.v4.new_notebook()
    cells = []

    intro_md = f"# {subtitle}\n\n{desc}\n\n{RUNTIME_NOTE}\n\n"
    if docstring:
        intro_md += f"---\n\n{docstring}\n"
    cells.append(nbf.v4.new_markdown_cell(intro_md))
    cells.append(nbf.v4.new_code_cell("%matplotlib inline"))

    for title, code in split_source(source):
        code = code.strip("\n")
        if not code:
            continue
        if title:
            cells.append(nbf.v4.new_markdown_cell(f"## {title}"))
        cells.append(nbf.v4.new_code_cell(code))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", out_path)


if __name__ == "__main__":
    for fname, (out_stem, subtitle, desc) in INTROS.items():
        py_path = os.path.join(HERE, fname)
        out_path = os.path.join(OUT_DIR, out_stem + ".ipynb")
        build_notebook(py_path, out_path, out_stem, subtitle, desc)
