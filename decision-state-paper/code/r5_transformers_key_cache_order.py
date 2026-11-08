"""What the cached keys are in Hugging Face transformers' Qwen3 / Qwen3.5 / Qwen3-Next attention (review
round 5, finding 12).

Reads library source from a transformers wheel without importing or running it (pass the wheel's path).
For each attention class it records the line of the key normalisation (k_norm), the rotary position
embedding (apply_rotary_pos_emb) and the cache write (past_key_values.update) in forward(), so their order
shows what the cache holds; and the DynamicCache lines that build one cache layer per config.layer_types
entry. The transformers version used for the paper's run is not recorded anywhere in this repository.

The wheel hash is given twice: of the LF form (this repository's rule for recorded input hashes) and of the
raw bytes, which is what the package index publishes.
Output: data/r5_transformers_key_cache_order.json
"""
import ast
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
OUT = PAPER / "data" / "r5_transformers_key_cache_order.json"
MODELS = ["qwen3", "qwen3_5", "qwen3_next"]


def call_name(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        base = f.value
        prefix = base.attr if isinstance(base, ast.Attribute) else (base.id if isinstance(base, ast.Name) else "")
        return f"{prefix}.{f.attr}"
    if isinstance(f, ast.Name):
        return f.id
    return ""


def attention_order(src):
    tree = ast.parse(src)
    lines = src.split("\n")
    out = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Attention")]:
        fwd = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
        if not fwd:
            continue
        rec = {}
        for node in ast.walk(fwd[0]):
            if isinstance(node, ast.Call):
                nm = call_name(node)
                for key, pat in (("k_norm", r"\.k_norm$"), ("apply_rotary_pos_emb", r"^apply_rotary_pos_emb$"),
                                 ("cache_update", r"past_key_values?\.update$")):
                    if re.search(pat, nm) and (key not in rec or node.lineno < rec[key]["line"]):
                        rec[key] = {"line": node.lineno, "code": lines[node.lineno - 1].strip()}
        if rec:
            out[cls.name] = rec
    return out


def main():
    wheel = Path(sys.argv[1])
    raw = wheel.read_bytes()
    z = zipfile.ZipFile(wheel)
    names = z.namelist()
    version = next((re.search(r"transformers-([^/]+)\.dist-info/", n).group(1) for n in names
                    if re.search(r"transformers-[^/]+\.dist-info/METADATA$", n)), None)
    models = {}
    for m in MODELS:
        path = f"transformers/models/{m}/modeling_{m}.py"
        if path in names:
            src = z.read(path).decode("utf-8").replace("\r\n", "\n")
            models[path] = attention_order(src)
    cu_path = "transformers/cache_utils.py"
    cu = z.read(cu_path).decode("utf-8").replace("\r\n", "\n").split("\n")
    tree = ast.parse("\n".join(cu))
    dyn = {}
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "DynamicCache":
            dyn = {"class_lines": [n.lineno, n.end_lineno],
                   "layer_types_lines": [{"line": i + 1, "code": cu[i].strip()} for i in range(n.lineno - 1, n.end_lineno)
                                         if "layer_types" in cu[i]]}
    result = {
        "LABEL": "Review round 5, finding 12. Library source read from the wheel, not imported; the run's transformers version is not recorded in this repository.",
        "wheel": {"file": wheel.name, "version": version,
                  "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                  "sha256_raw": hashlib.sha256(raw).hexdigest()},
        "attention_forward_order": models,
        "DynamicCache": dyn,
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(REPO).as_posix())
    print("  version", version)
    for p, cls in models.items():
        for c, r in cls.items():
            print("  ", p.split("/")[-1], c, {k: v["line"] for k, v in r.items()})
    print("  DynamicCache", dyn.get("class_lines"), [x["line"] for x in dyn.get("layer_types_lines", [])])


if __name__ == "__main__":
    main()
