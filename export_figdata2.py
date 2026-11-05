# -*- coding: utf-8 -*-
"""一次性导出 Fig.7(收敛图)/Fig.8(权衡图) 所需的全部数据。只读 log，不碰 ckpt。"""
import os, re, csv, glob, shutil, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(ROOT, 'figdata_pack')
OUT  = os.path.join(PACK, 'figdata_out')
if os.path.isdir(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
MAX_EPOCH = 150

RUNS = [
    ('HPE-PEFT',   'experiments/config/peft_hpe_seg_perlayer/hpe_seg_perlayer_v2', True),
    ('Adapter',    'experiments/shapenet_part_seg_adapter/cfgs/default', True),
    ('LoRA',       'experiments/shapenet_part_seg_lora/cfgs/default', True),
    ('VPT',        'experiments/shapenet_part_seg_prompt/cfgs/default', True),
    ('Shared',     'experiments/shapenet_part_seg_shared/cfgs/peft_hpe_seg_shared', True),
    ('PointLoRA',  'experiments/shapenet_part_seg_pointlora/cfgs/shapenet_part_seg_pointlora', True),
    ('Point-PEFT', 'experiments/shapenet_part_seg_pointpeft/cfgs/shapenet_part_seg_pointpeft', True),
    ('FullFT-LEGACY', 'experiments/shapenet_part_seg_ft/cfgs/shapenet_part_seg_ft', False),
]

RE_TRAIN = re.compile(r"\[Training\]\s+EPOCH:\s*(\d+)\s+Losses\s*=\s*\['([^']+)',\s*'([^']+)'\]\s+lr\s*=\s*([0-9.eE+-]+)")
RE_VAL   = re.compile(r"\[Validation\]\s+EPOCH:\s*(\d+)\s+mIoU\s*=\s*([0-9.]+)\s+Acc\s*=\s*([0-9.]+)")

def parse_run(rel):
    d = os.path.join(ROOT, rel)
    logs = sorted(glob.glob(os.path.join(d, '*.log')))
    recs, dropped, per_log = {}, [], []
    for lg in logs:
        n_tr = n_val = 0
        for ln in open(lg, errors='ignore'):
            m = RE_TRAIN.search(ln)
            if m:
                e = int(m.group(1))
                if e > MAX_EPOCH: dropped.append(('train', e)); continue
                n_tr += 1
                r = recs.setdefault(e, {}); r['train_loss'] = m.group(2); r['lr'] = m.group(4)
                continue
            m = RE_VAL.search(ln)
            if m:
                e = int(m.group(1))
                if e > MAX_EPOCH: dropped.append(('val', e)); continue
                n_val += 1
                r = recs.setdefault(e, {}); r['val_miou'] = m.group(2); r['val_acc'] = m.group(3)
        per_log.append((os.path.basename(lg), n_tr, n_val))
    pts = [(e, recs[e]) for e in sorted(recs) if 'val_miou' in recs[e]]
    return logs, per_log, pts, sorted(set(dropped))

def write_csv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)

all_rows = []
print("%-13s %-5s %-5s %-14s %-9s %-9s %s" % ('method','logs','pts','epoch range','first','last','best'))
for label, rel, main in RUNS:
    logs, per_log, pts, dropped = parse_run(rel)
    rows = [[label, e, r.get('train_loss',''), r['val_miou'], r.get('val_acc',''), r.get('lr','')] for e, r in pts]
    target = OUT if main else os.path.join(OUT, 'legacy')
    os.makedirs(target, exist_ok=True)
    write_csv(os.path.join(target, 'curve_%s.csv' % label.replace(' ','_')), rows,
              ['method','epoch','train_loss','val_miou','val_acc','lr'])
    if main: all_rows += rows
    best = max(pts, key=lambda t: float(t[1]['val_miou'])) if pts else None
    print("%-13s %-5d %-5d %-14s %-9s %-9s %s" % (
        label, len(logs), len(pts),
        ('%d..%d' % (pts[0][0], pts[-1][0])) if pts else 'NO DATA',
        pts[0][1]['val_miou'] if pts else '-', pts[-1][1]['val_miou'] if pts else '-',
        ('%s @ep%d' % (best[1]['val_miou'], best[0])) if best else '-'))
    for name, n_tr, n_val in per_log:
        print("      log %-26s train行=%-4d val行=%d" % (name, n_tr, n_val))
    if dropped: print("      [丢弃] epoch>%d 的伪点: %s" % (MAX_EPOCH, dropped))

write_csv(os.path.join(OUT, 'curves_all.csv'), all_rows,
          ['method','epoch','train_loss','val_miou','val_acc','lr'])
print("\nmerged -> curves_all.csv (%d 行)" % len(all_rows))

for f in ['eval8_val.csv', 'eval8_test.csv']:
    src = os.path.join(ROOT, f)
    if os.path.exists(src): shutil.copy(src, OUT); print("copied %s (%d bytes)" % (f, os.path.getsize(src)))
    else: print("!! 缺 %s" % f)

readme = """图数据包说明（export_figdata2.py 生成）
================================================================
[1] curves_all.csv / curve_<方法>.csv  —— Fig.7 收敛图
   列: method, epoch, train_loss, val_miou, val_acc, lr
   * 数据源: 训练日志的 [Validation] 行; val_miou 是 ShapeNetPart 验证集 Part-mIoU (0~1)
   * 采样密度天然不同(HPE-PEFT/Adapter/LoRA/VPT 的日志只记了 14~22 个 val 点;
     Shared/PointLoRA/Point-PEFT 每 epoch 都记, 150 点)。按原样折线画, 不要插值对齐。
   * legacy/curve_FullFT-LEGACY.csv 是旧评估协议(val mIoU 约 0.90), 与上面 7 条不同尺子, 禁止同图。
   * 已剔除日志里夹带的 epoch>150 伪点。

[2] eval8_val.csv / eval8_test.csv  —— Fig.8 权衡图
   列: method, params_M, ratio_pct, part_miou, inst_miou, cls_miou, acc, logged_best_val, delta_part_minus_logged
   * part_miou / acc 是百分数(81.32 表示 81.32%), params_M 单位百万, ratio_pct 是百分比。
   * 论文 Table VII 用 test 那份; val 那份仅供对照。
================================================================
"""
open(os.path.join(OUT, 'README.txt'), 'w').write(readme)

zpath = os.path.join(PACK, 'figdata.zip')
if os.path.exists(zpath): os.remove(zpath)
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for r_, _, fs in os.walk(OUT):
        for fn in fs:
            p = os.path.join(r_, fn)
            z.write(p, os.path.relpath(p, OUT))
print("\nzip: %s  (%d bytes)" % (zpath, os.path.getsize(zpath)))
with zipfile.ZipFile(zpath) as z:
    for i in z.infolist(): print("   %-40s %6d" % (i.filename, i.file_size))
