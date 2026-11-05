import os, re, glob, csv, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "figdata_pack", "figdata")
ZIP  = os.path.join(ROOT, "figdata_pack", "figdata.zip")

RUNS = [
 ("shapenet_part_seg_shared",    "experiments/shapenet_part_seg_shared/cfgs/peft_hpe_seg_shared", 81.09),
 ("shapenet_part_seg_adapter",   "experiments/shapenet_part_seg_adapter/cfgs/default", 81.37),
 ("shapenet_part_seg_lora",      "experiments/shapenet_part_seg_lora/cfgs/default", 78.42),
 ("shapenet_part_seg_prompt",    "experiments/shapenet_part_seg_prompt/cfgs/default", 80.36),
 ("shapenet_part_seg_pointlora", "experiments/shapenet_part_seg_pointlora/cfgs/shapenet_part_seg_pointlora", 80.47),
 ("shapenet_part_seg_pointpeft", "experiments/shapenet_part_seg_pointpeft/cfgs/shapenet_part_seg_pointpeft", 80.17),
 ("shapenet_part_seg_hpe_v2",    "experiments/config/peft_hpe_seg_perlayer/hpe_seg_perlayer_v2", 80.87),
]
EXTRA = ["eval8_val.csv", "eval8_test.csv"]

NUM   = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
RE_EP = re.compile(r"epoch[^0-9]{0,16}(" + NUM + ")", re.I)
RE_MI = re.compile(r"(?:miou|mean[ _]?iou)[^0-9\-+]{0,16}(" + NUM + ")", re.I)
RE_LS = re.compile(r"loss[^0-9\-+]{0,16}(" + NUM + ")", re.I)


def newest_log(d):
    logs = glob.glob(os.path.join(d, "**", "*.log"), recursive=True)
    return max(logs, key=os.path.getmtime) if logs else None


def parse(log):
    text = open(log, encoding="utf-8", errors="ignore").read()
    lines = re.split(r"\r\n|\r|\n", text)
    hits = [l for l in lines if RE_MI.search(l)]
    seen = {}
    for k, l in enumerate(hits):
        m = RE_EP.search(l)
        ep = int(float(m.group(1))) if m else k + 1
        v = float(RE_MI.search(l).group(1))
        if v > 1.5:
            v = v / 100.0
        ls = RE_LS.search(l)
        seen[ep] = (ep, float(ls.group(1)) if ls else None, v)
    return len(hits), [seen[e] for e in sorted(seen)]


os.makedirs(OUT, exist_ok=True)
warn = []
for name, reld, exp in RUNS:
    d = os.path.join(ROOT, reld)
    log = newest_log(d) if os.path.isdir(d) else None
    if log is None:
        print("WARN", name, "-> 目录或日志缺失:", reld)
        warn.append(name)
        continue
    nh, rows = parse(log)
    if not rows:
        print("WARN", name, "-> 没解析出数据，含 miou 的行数:", nh)
        warn.append(name)
        continue
    with open(os.path.join(OUT, name + ".csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_miou"])
        for ep, ls, v in rows:
            w.writerow([ep, "" if ls is None else round(ls, 6), round(v, 6)])
    vals = [r[2] for r in rows]
    ok = "OK" if abs(max(vals) * 100 - exp) <= 0.5 else "WARN exp=%.2f" % exp
    if not ok.startswith("OK"):
        warn.append(name)
    print("%-30s %-24s ep=%3d best=%6.2f final=%6.2f  %s"
          % (name, os.path.basename(log), len(rows),
             max(vals) * 100, vals[-1] * 100, ok))

for f in EXTRA:
    s = os.path.join(ROOT, f)
    if os.path.isfile(s):
        shutil.copyfile(s, os.path.join(OUT, f))
        print("extra:", f)

if os.path.exists(ZIP):
    os.remove(ZIP)
shutil.make_archive(ZIP[:-4], "zip", OUT)
print("zip:", ZIP, os.path.getsize(ZIP), "bytes")
print("WARN list:", warn if warn else "none")
