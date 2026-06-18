#!/usr/bin/env python3
"""
Regenerate bregman-lab-suite-standalone.html from index.html + src/.

The standalone is index.html with (a) src/styles.css inlined into a <style> block and
(b) the ES-module graph (src/main.js and its imports) concatenated into ONE non-module
<script>, with every `import ...`/`export ...` line stripped. Run after editing any src file:

    python3 build_standalone.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# modules in dependency order (leaves first)
MODULES = [
    SRC / "core" / "math.js",
    SRC / "core" / "dom.js",
    SRC / "tasks" / "task1-dual-manifold.js",
    SRC / "tasks" / "task2-theorem-live.js",
    SRC / "tasks" / "task3-admissibility.js",
    SRC / "main.js",
]

IMPORT_RE = re.compile(r"^\s*import\s.*?;?\s*$")
EXPORT_BLOCK_RE = re.compile(r"^\s*export\s*\{.*?\};?\s*$")


def strip_module(text: str) -> str:
    out = []
    for line in text.splitlines():
        if IMPORT_RE.match(line) or EXPORT_BLOCK_RE.match(line):
            continue                                   # drop import / `export { ... };`
        line = re.sub(r"^(\s*)export\s+function\s", r"\1function ", line)
        line = re.sub(r"^(\s*)export\s+const\s", r"\1const ", line)
        line = re.sub(r"^(\s*)export\s+default\s", r"\1", line)
        out.append(line)
    return "\n".join(out)


def main():
    html = (ROOT / "index.html").read_text()
    css = (SRC / "styles.css").read_text()

    bundle = "\n\n".join(
        f"/* ===== {m.relative_to(ROOT)} ===== */\n{strip_module(m.read_text())}"
        for m in MODULES
    )

    # inline stylesheet / module graph. NOTE: use FUNCTION replacements (lambda) — a plain string
    # replacement in re.sub processes backslash escapes (\n, \1, \\ ...), which would corrupt JS
    # string literals like '<?xml ...?>\n' (turning \n into a real newline → unterminated string).
    # A function replacement is inserted verbatim, no escape processing.
    html, n_css = re.subn(
        r'<link rel="stylesheet" href="\./src/styles\.css"\s*/?>',
        lambda m: f"<style>\n{css}\n</style>",
        html,
    )
    html, n_js = re.subn(
        r'<script type="module" src="\./src/main\.js"></script>',
        lambda m: f"<script>\n{bundle}\n</script>",
        html,
    )
    if n_css != 1 or n_js != 1:
        raise SystemExit(f"template markers not found (css={n_css}, js={n_js}) — check index.html")

    out = ROOT / "bregman-lab-suite-standalone.html"
    out.write_text(html)
    print(f"wrote {out.name}  ({len(html)//1024} KB, {len(MODULES)} modules inlined)")


if __name__ == "__main__":
    main()
