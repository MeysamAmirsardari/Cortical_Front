"""Validate the eSTRF-response notebook: JSON well-formed, every code
cell parses as Python, no obvious bugs."""
import ast
import json
from pathlib import Path

NB = Path("/Users/eminent/Projects/Cortical_Front/estrf_responses_along_utterances.ipynb")

with open(NB) as f:
    nb = json.load(f)

print(f"Loaded {NB.name}: {len(nb['cells'])} cells, "
      f"nbformat={nb['nbformat']}.{nb['nbformat_minor']}")

n_code = n_md = 0
errors = []
for i, cell in enumerate(nb["cells"]):
    ctype = cell["cell_type"]
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if ctype == "code":
        n_code += 1
        # Strip Colab/IPython magic so ast.parse works.
        clean = []
        for line in src.splitlines():
            ls = line.lstrip()
            if ls.startswith("%") or ls.startswith("!"):
                indent = line[: len(line) - len(ls)]
                clean.append(f"{indent}pass  # MAGIC: {line.strip()}")
            else:
                clean.append(line)
        clean_src = "\n".join(clean)
        try:
            ast.parse(clean_src)
        except SyntaxError as e:
            errors.append(f"Cell #{i} ({ctype}): SyntaxError {e}")
    elif ctype == "markdown":
        n_md += 1
    else:
        errors.append(f"Cell #{i}: unknown type {ctype}")

print(f"  code cells: {n_code}")
print(f"  md   cells: {n_md}")
if errors:
    print("\nERRORS:")
    for e in errors:
        print(" ", e)
else:
    print("\nAll cells parse cleanly.")
