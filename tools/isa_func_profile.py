import re
import sys
from collections import Counter

path = sys.argv[1]
needle = sys.argv[2] if len(sys.argv) > 2 else ""

src = open(path, encoding="utf-8", errors="replace").read()
funcs = re.split(r"\n(?=_Z)", src)
rows = []
for f in funcs:
    m = re.match(r"(_Z\S+)", f)
    if not m:
        continue
    name = m.group(1)
    if needle and needle not in name:
        continue
    if "device_stub" in name:
        continue
    vg = [int(x) for x in re.findall(r"\$vgpr(\d+)", f)]
    lines = [l.strip() for l in f.splitlines()]
    insts = [l for l in lines if l and not l.startswith(".") and not l.startswith("_Z")
             and not l.endswith(":") and not l.startswith(";") and not l.startswith("//")]
    scratch = sum(1 for l in insts if "scratch" in l)
    ops = Counter(l.split()[0].rstrip("_") for l in insts if l.split())
    rows.append((name[:70], (max(vg) + 1) if vg else 0, len(insts), scratch,
                 dict(ops.most_common(6))))
rows.sort(key=lambda r: -r[2])
print("func                                                          maxvgpr  insts  scratch  top-ops")
for name, v, n, s, ops in rows[:14]:
    print("%-60s %7d %6d %7d  %s" % (name, v, n, s, str(ops)[:90]))
