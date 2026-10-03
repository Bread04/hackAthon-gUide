"""Compare pinned versions in the project template against the latest on PyPI.

Usage (from the repo root):
    python tools/check_versions.py            # print a report, exit 0
    python tools/check_versions.py --strict   # exit 1 if any pin is behind
    python tools/check_versions.py --markdown # report as a Markdown table

A pin being behind is not an error by itself: bump it only after
`python tools/smoke_test.py` passes with the new version.
"""
import json
import re
import sys
import urllib.request

REQ = "08-datathon-handbook/PROJECT_TEMPLATE/requirements.txt"


def latest(pkg):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=20) as r:
        return json.load(r)["info"]["version"]


def key(v):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v))


pins = []
for line in open(REQ, encoding="utf-8"):
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
    if m:
        pins.append(m.groups())

rows, behind = [], 0
for pkg, pinned in pins:
    try:
        new = latest(pkg)
        status = "ok" if key(new) <= key(pinned) else "BEHIND"
    except Exception as e:  # network errors are reported, not fatal
        new, status = "?", f"error: {e.__class__.__name__}"
    behind += status == "BEHIND"
    rows.append((pkg, pinned, new, status))

if "--markdown" in sys.argv:
    print("| Package | Pinned | Latest on PyPI | Status |\n| --- | --- | --- | --- |")
    for r in rows:
        print("| " + " | ".join(r) + " |")
else:
    for r in rows:
        print(f"{r[0]:16} pinned {r[1]:10} latest {r[2]:10} {r[3]}")
print(f"\n{behind} of {len(rows)} pins are behind PyPI.")
sys.exit(1 if ("--strict" in sys.argv and behind) else 0)
