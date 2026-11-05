#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_all8.py —— 路线 B：8 个 ShapeNetPart 部件分割 run 的六列同源数据
指标与 eval_shapenet_inst_miou.py 完全一致：
  Part-mIoU : 50 个部件类全局累积 IoU 再平均
  Inst-mIoU : 逐实例 IoU 再对所有实例平均
  Cls-mIoU  : 16 大类各自实例平均后再对 16 类平均
  Acc       : 点级准确率
切分默认 val（与训练时模型选择一致），dLog = 本次 Part-mIoU - 日志 best val mIoU。
"""
import os, sys, csv, random, argparse
from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from utils.config import get_config
from tools import builder
from pointnet2_ops import pointnet2_utils

RANGE_TO_NAME = {
    (0, 1, 2, 3): 'Airplane', (4, 5): 'Bag', (6, 7): 'Cap',
    (8, 9, 10, 11): 'Car', (12, 13, 14, 15): 'Chair',
    (16, 17, 18): 'Earphone', (19, 20, 21): 'Guitar',
    (22, 23): 'Knife', (24, 25, 26, 27): 'Lamp', (28, 29): 'Laptop',
    (30, 31, 32, 33, 34, 35): 'Motorbike', (36, 37): 'Mug',
    (38, 39, 40): 'Pistol', (41, 42, 43): 'Rocket',
    (44, 45, 46): 'Skateboard', (47, 48, 49): 'Table',
}

# (显示名, run 目录, cfg 名)
RUNS = [
    ("FullFT",     "shapenet_part_seg_ft",        "shapenet_part_seg_ft"),
    ("HPE-PEFT",   "shapenet_part_seg",           "peft_hpe_seg_perlayer"),
    ("Adapter",    "shapenet_part_seg_adapter",   "default"),
    ("Shared HPE", "shapenet_part_seg_shared",    "peft_hpe_seg_shared"),
    ("LoRA",       "shapenet_part_seg_lora",      "default"),
    ("VPT",        "shapenet_part_seg_prompt",    "default"),
    ("PointLoRA",  "shapenet_part_seg_pointlora", "shapenet_part_seg_pointlora"),
    ("Point-PEFT", "shapenet_part_seg_pointpeft", "shapenet_part_seg_pointpeft"),
]

LOGGED = {"FullFT": 89.98, "HPE-PEFT": 87.72, "Adapter": 81.37, "Shared HPE": 81.09,
          "LoRA": 78.42, "VPT": 80.36, "PointLoRA": 80.47, "Point-PEFT": 80.17}

FULLFT_M = 22.7688


def build_ns(cfg_path, exp_dir, ckpt):
    return SimpleNamespace(
        config=cfg_path, exp_name=os.path.basename(exp_dir), ckpts=ckpt,
        log_name='eval_all8', distributed=False, local_rank=0,
        use_gpu=True, num_workers=0, resume=False, seed=0, deterministic=False,
        launcher='none', world_size=1, test=True, experiment_path=exp_dir,
    )


def evaluate(disp, run, cfg_name, split, seed=0, ckpt_name='ckpt-best'):
    exp_dir  = os.path.join('experiments', run, 'cfgs', cfg_name)
    cfg_path = os.path.join(exp_dir, 'config.yaml')
    ckpt     = os.path.join(exp_dir, ckpt_name + '.pth')
    assert os.path.exists(cfg_path), 'missing config: ' + cfg_path
    assert os.path.exists(ckpt),     'missing ckpt:   ' + ckpt

    scratch = os.path.join('experiments', '_eval_all8', run)
    os.makedirs(scratch, exist_ok=True)
    ns = build_ns(cfg_path, scratch, ckpt)

    # 固定随机种子 -> 8 个方法看到同一批随机采样点，公平可比
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    config = get_config(ns)
    npoints = config.npoints
    split_cfg = getattr(config.dataset, split)
    split_cfg.others['bs'] = config.total_bs
    _, loader = builder.dataset_builder(ns, split_cfg)
    ds = loader.dataset

    ranges = {}
    for (cid, pid), gid in ds.cat_local_to_global.items():
        ranges.setdefault(cid, []).append(gid)

    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt)
    model = model.cuda().eval()
    # Point-PEFT 的 point-prior bank 是 persistent=False 的 buffer，不在 ckpt 里；
    # 训练时由 model.load_model_from_ckpt() 负责加载，评估路径需手动触发一次。
    if hasattr(model, '_load_bank') and getattr(model, 'bank_path', None):
        if getattr(model, 'bank', None) is None or model.bank.numel() == 0:
            model._load_bank()
            print('    [bank] loaded, shape = %s' % (tuple(model.bank.shape),))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print('    [npoints] %d   [trainable] %.4f M' % (npoints, n_train))

    all_pred, all_gt, all_cat = [], [], []
    with torch.no_grad():
        for taxonomy_ids, model_ids, data in tqdm(loader, desc='  Infer ' + disp, ncols=80):
            points    = data[0].cuda()
            seg_label = data[1].cuda()
            fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
            points = pointnet2_utils.gather_operation(
                points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            seg_label = torch.gather(seg_label, 1, fps_idx.long())
            pred = model(points).argmax(dim=1)
            all_pred.append(pred.detach().cpu())
            all_gt.append(seg_label.detach().cpu())
            for c in taxonomy_ids:
                all_cat.append(c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else c)

    all_pred = torch.cat(all_pred, 0).numpy()
    all_gt   = torch.cat(all_gt, 0).numpy()

    # ---- Part-mIoU (50 类) + 点级 Acc ----
    ious = []
    for p in range(50):
        inter = ((all_pred == p) & (all_gt == p)).sum()
        union = ((all_pred == p) | (all_gt == p)).sum()
        if union > 0:
            ious.append(inter / float(union))
    part_miou = float(np.mean(ious)) * 100.
    acc = float((all_pred == all_gt).mean()) * 100.

    # ---- Inst / Cls mIoU（官方协议）----
    inst_ious = []
    class_ious = {cid: [] for cid in ranges}
    for i in range(all_pred.shape[0]):
        segp, segl, cid = all_pred[i], all_gt[i], all_cat[i]
        if cid not in ranges:
            continue
        vals = []
        for l in ranges[cid]:
            g = (segl == l).sum()
            if g == 0:
                vals.append(1.0)
            else:
                vals.append(((segl == l) & (segp == l)).sum() /
                            float(((segl == l) | (segp == l)).sum()))
        v = float(np.mean(vals))
        inst_ious.append(v)
        class_ious[cid].append(v)
    inst_miou = float(np.mean(inst_ious)) * 100.
    cls_miou  = float(np.mean([np.mean(v) for v in class_ious.values() if len(v)])) * 100.

    del model
    torch.cuda.empty_cache()
    return dict(method=disp, part=part_miou, inst=inst_miou, cls=cls_miou,
                acc=acc, params=n_train, logged=LOGGED.get(disp, float('nan')),
                n_inst=len(inst_ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--csv', default='eval8_results.csv')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt-name', default='ckpt-best')
    opt = ap.parse_args()

    todo = [r for r in RUNS if opt.only is None or r[0] in opt.only]
    print('=' * 88)
    print('split = %s   methods = %d' % (opt.split, len(todo)))
    print('=' * 88)

    rows = []
    for disp, run, cfg in todo:
        print('\n>>> %s   (%s / %s)' % (disp, run, cfg))
        try:
            rows.append(evaluate(disp, run, cfg, opt.split, opt.seed, opt.ckpt_name))
        except Exception as e:
            print('    [FAILED] %s: %s' % (disp, e))
            import traceback; traceback.print_exc()

    with open(opt.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'params_M', 'ratio_pct', 'part_miou', 'inst_miou',
                    'cls_miou', 'acc', 'logged_best_val', 'delta_part_minus_logged'])
        for r in rows:
            w.writerow([r['method'], '%.2f' % r['params'], '%.2f' % (r['params'] / FULLFT_M * 100),
                        '%.2f' % r['part'], '%.2f' % r['inst'], '%.2f' % r['cls'],
                        '%.2f' % r['acc'], '%.2f' % r['logged'], '%+.2f' % (r['part'] - r['logged'])])

    print('\n' + '=' * 88)
    print('%-12s %8s %8s %8s %8s %8s %9s' % ('Method', 'Param/M', 'Part', 'Inst', 'Cls', 'Acc', 'dLog'))
    print('-' * 88)
    for r in rows:
        print('%-12s %8.2f %8.2f %8.2f %8.2f %8.2f %+9.2f'
              % (r['method'], r['params'], r['part'], r['inst'], r['cls'],
                 r['acc'], r['part'] - r['logged']))
    print('=' * 88)

    print('\n%% ==== LaTeX rows (Method & Param & Ratio & Part & Inst & Cls & Acc) ====')
    for r in rows:
        nm = r['method']
        nm = r'\textbf{HPE-PEFT (Ours)}' if nm == 'HPE-PEFT' else nm
        print('%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\'
              % (nm, r['params'], r['params'] / FULLFT_M * 100,
                 r['part'], r['inst'], r['cls'], r['acc']))
    print('\n[CSV]  %s' % os.path.abspath(opt.csv))
    print('[NOTE] dLog 是所有结论的判据：接近 0 => 日志=标准 Part-mIoU；')
    print('       明显为负且 Inst 列接近日志值 => 日志其实是 Inst-mIoU，请把结果发我。')


if __name__ == '__main__':
    main()
