import os, sys, argparse
from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.config import get_config
from tools import builder
from pointnet2_ops import pointnet2_utils

# 标准 16 类的全局部件标签范围 -> 类别名（仅用于显示，统计不依赖它）
RANGE_TO_NAME = {
    (0, 1, 2, 3): 'Airplane', (4, 5): 'Bag', (6, 7): 'Cap',
    (8, 9, 10, 11): 'Car', (12, 13, 14, 15): 'Chair',
    (16, 17, 18): 'Earphone', (19, 20, 21): 'Guitar',
    (22, 23): 'Knife', (24, 25, 26, 27): 'Lamp', (28, 29): 'Laptop',
    (30, 31, 32, 33, 34, 35): 'Motorbike', (36, 37): 'Mug',
    (38, 39, 40): 'Pistol', (41, 42, 43): 'Rocket',
    (44, 45, 46): 'Skateboard', (47, 48, 49): 'Table',
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='cfgs/shapenet_part_seg_export.yaml')
    p.add_argument('--exp_name', type=str, default='inst_miou_eval')
    p.add_argument('--ckpt', type=str, default=None)
    return p.parse_args()

def norm_cat(c):
    if isinstance(c, (bytes, bytearray)):
        c = c.decode('utf-8')
    return c

def main():
    args = parse_args()
    if args.ckpt is None:
        args.ckpt = f'./experiments/{args.exp_name}/cfgs/{args.exp_name}/ckpt-best.pth'

    ns = SimpleNamespace(
        config=args.config, exp_name=args.exp_name, ckpts=args.ckpt,
        log_name='inst_miou_eval', distributed=False, local_rank=0,
        use_gpu=True, num_workers=0, resume=False, seed=0, deterministic=False,
        launcher='none', world_size=1, test=True,
        experiment_path=f'./experiments/{args.exp_name}/cfgs/{args.exp_name}',
    )
    config = get_config(ns)
    npoints = config.npoints

    # 固定随机种子：消除 ShapeNetPart 随机点采样的波动，保证可复现
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print('[npoints]', npoints)

    config.dataset.test.others["bs"] = config.total_bs
    _, test_dataloader = builder.dataset_builder(ns, config.dataset.test)
    ds = test_dataloader.dataset

    # 从数据集自带映射构建 cat_id -> 全局部件标签列表（不依赖 synset 名）
    ranges = {}
    for (cid, pid), gid in ds.cat_local_to_global.items():
        ranges.setdefault(cid, []).append(gid)
    print('\n[cat_id -> 部件标签范围 -> 类别名]')
    for cid in sorted(ranges):
        parts = tuple(sorted(ranges[cid]))
        print('  %-10s %s -> %s' % (cid, parts, RANGE_TO_NAME.get(parts, '?')))

    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpt)
    model = model.cuda().eval()

    all_pred, all_gt, all_cat = [], [], []
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader, desc='Infer')):
            points = data[0].cuda()
            seg_label = data[1].cuda()
            fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
            points = pointnet2_utils.gather_operation(
                points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            seg_label = torch.gather(seg_label, 1, fps_idx.long())
            logits = model(points)
            pred = logits.argmax(dim=1)
            all_pred.append(pred.detach().cpu())
            all_gt.append(seg_label.detach().cpu())
            for c in taxonomy_ids:
                all_cat.append(norm_cat(c))

    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_gt = torch.cat(all_gt, dim=0).numpy()
    print('\n[samples]', all_pred.shape)

    # 全局 Part-mIoU（50 类）与点级 Acc（验证）
    ious = []
    for p in range(50):
        inter = ((all_pred == p) & (all_gt == p)).sum()
        union = ((all_pred == p) | (all_gt == p)).sum()
        if union > 0:
            ious.append(inter / union)
    print('[Global] Part-mIoU = %.4f  Acc = %.4f' % (float(np.mean(ious)),
                                                     float((all_pred == all_gt).mean() * 100.)))

    # instance / class mIoU（官方协议）
    inst_ious = []
    class_ious = {cid: [] for cid in ranges}
    for i in range(all_pred.shape[0]):
        segp = all_pred[i]
        segl = all_gt[i]
        cid = all_cat[i]
        if cid not in ranges:
            continue
        part_ids = ranges[cid]
        part_ious = [0.0] * len(part_ids)
        for k, l in enumerate(part_ids):
            if np.sum(segl == l) == 0:
                part_ious[k] = 1.0
            else:
                part_ious[k] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
        inst_iou = float(np.mean(part_ious))
        inst_ious.append(inst_iou)
        class_ious[cid].append(inst_iou)

    print('\n[Per-class mIoU]')
    for cid in sorted(class_ious):
        parts = tuple(sorted(ranges[cid]))
        name = RANGE_TO_NAME.get(parts, cid)
        vals = class_ious[cid]
        print('  %-14s %s  %.4f  (n=%d)' % (name, cid, float(np.mean(vals)), len(vals)))

    instance_miou = float(np.mean(inst_ious))
    class_miou = float(np.mean([np.mean(v) for v in class_ious.values()]))
    print('\n[RESULT] instance mIoU = %.4f  class mIoU = %.4f' % (instance_miou, class_miou))

if __name__ == '__main__':
    main()
