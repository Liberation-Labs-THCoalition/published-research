"""Patch llm.py's ollama path to stop discarding reasoning output.

Run ON the target host so no shell quoting layer touches it. Uses chr(10)
rather than an escape sequence, because heredoc transport has silently eaten
one level of backslash twice tonight and turned a string literal into a syntax
error both times.

THE BUG BEING FIXED: _ollama returns resp.json().get("response", ""). Reasoning
models (qwen3.x) emit their analysis as `thinking` and only the final answer as
`response`. A reviewer that spends its budget reasoning therefore returns an
EMPTY STRING, and Agni records a blank review as though the model had nothing
to say -- a silent drop indistinguishable from a genuine non-finding.
"""
import sys

PATH = "/tmp/agni/llm.py"
NL = chr(10)

src = open(PATH).read()

old = (
    '    resp = requests.post(f"{url}/api/generate", json=body, timeout=timeout)' + NL +
    '    if resp.status_code == 200:' + NL +
    '        return resp.json().get("response", "")'
)

new = (
    '    resp = requests.post(f"{url}/api/generate", json=body, timeout=timeout)' + NL +
    '    if resp.status_code == 200:' + NL +
    '        j = resp.json()' + NL +
    '        out = j.get("response", "") or ""' + NL +
    '        think = j.get("thinking", "") or ""' + NL +
    '        if think and not out.strip():' + NL +
    '            return think' + NL +
    '        if think:' + NL +
    '            return think + chr(10) + chr(10) + out' + NL +
    '        return out'
)

if old not in src:
    print("PATCH TARGET NOT FOUND -- file may already be patched or altered")
    sys.exit(2)

open(PATH, "w").write(src.replace(old, new))

# prove it parses rather than assuming it does
import py_compile
py_compile.compile(PATH, doraise=True)
print("patched and compiles clean:", PATH)
