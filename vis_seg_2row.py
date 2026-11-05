import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

sys.path = [p for p in sys.path if 'hpenext' not in p]
sys.path.insert(0, '.')

from tools import builder
from utils.config import get_config

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PART_CMAP = plt.cm.tab20
PART_CMAP2 = plt.cm.tab20b

def get_part_color(part_id):
    if part_id < 20:
        return PART_CMAP(part_id / 20)
    else:
        return PART_CMAP2((part_id - 20) / 20)

def plot_pts(ax, pts, labels, elev=30, azim=120, title=''):
    colors = [get_part_color(int(l)) for l in labels]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=1.5, alpha=0.85, edgecolors='none')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    mr = max(abs(pts).max(), 0.5)
    ax.set_xlim(-mr, mr); ax.set_ylim(-mr, mr); ax.set_zlim(-mr, mr)

def main():
    class Args:
        config = 'cfgs/shapenet_part_seg.yaml'
        ckpts = 'experiments/shapenet_part_seg/cfgs/peft_hpe_seg_perlayer/ckpt-best.pth'
        launcher = 'none'
        local_rank = 0
        test = True
        use_gpu = torch.cuda.is_available()
        distributed = False
        num_workers = 4
        seed = 0
        deterministic = False
        sync_bn = False
        experiment_path = './experiments/vis_seg'
        tfboard_path = './experiments/vis_seg/tfboard'
        log_name = 'vis_seg'
        val_freq = 1
        vote = False
        resume = False
        finetune_model = False
        scratch_model = False
        mode = None
        way = -1
        shot = -1
        fold = -1
        start_ckpts = None
        loss = 'cd1'

    args = Args()
    print("=== 加载配置 ===")
    config = get_config(args, logger=None)
    config.dataset.test.others.bs = 1

    print("=== 加载模型 ===")
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=None)
    base_model.cuda()
    base_model.eval()
    torch.set_grad_enabled(False)

    print("=== 加载测试数据 ===")
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    vis_categories = {
        'Airplane': '02691156',
        'Chair': '03001627',
        'Guitar': '03467517',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Pistol': '03948459',
        'Table': '04379243',
    }

    samples_by_category = {name: [] for name in vis_categories}
    npoints = config.npoints
    from pointnet2_ops import pointnet2_utils

    print("=== 推理测试集 ===")
    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        points = data[0].cuda()
        seg_label = data[1].cuda()

        fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
        points = pointnet2_utils.gather_operation(
            points.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2).contiguous()
        seg_label = torch.gather(seg_label, 1, fps_idx.long())

        logits = base_model(points)
        pred = logits.argmax(dim=1)

        tax_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]
        for cat_name, cat_id in vis_categories.items():
            if tax_id == cat_id and len(samples_by_category[cat_name]) < 1:
                samples_by_category[cat_name].append({
                    'points': points[0].cpu().numpy(),
                    'pred': pred[0].cpu().numpy(),
                    'gt': seg_label[0].cpu().numpy(),
                    'model_id': str(model_ids[0]),
                })

        if idx % 50 == 0:
            print(f"  Processed {idx}...")

        if sum(len(v) for v in samples_by_category.values()) >= len(vis_categories) and idx > 100:
            print("  (样本收集完毕)")
            break

    # ===== 生成 2行×3列 大图 =====
    out_dir = 'visualization/shapenet_part_seg_hpe/paper_figures'
    os.makedirs(out_dir, exist_ok=True)

    views = [
        (30, 120, 'Side View'),
        (30, 30,  'Front View'),
        (15, 210, 'Top View'),
    ]

    print("\n=== 生成 2行×3列 对比图 ===")
    for cat_name, samples in samples_by_category.items():
        if not samples:
            continue

        sample = samples[0]
        pts = sample['points']
        gt_labels = sample['gt']
        pred_labels = sample['pred']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                                 subplot_kw={'projection': '3d'})

        for col, (elev, azim, view_name) in enumerate(views):
            # 第0行：Ground Truth
            ax = axes[0, col]
            plot_pts(ax, pts, gt_labels, elev=elev, azim=azim,
                     title=f'Ground Truth - {view_name}')

            # 第1行：HPE-PEFT
            ax = axes[1, col]
            plot_pts(ax, pts, pred_labels, elev=elev, azim=azim,
                     title=f'HPE-PEFT - {view_name}')

        fig.suptitle(f'{cat_name}', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout(pad=1.5)
        fname = f'{out_dir}/{cat_name}_gt_hpepeft.png'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {fname}")

    print(f"\n✅ 完成！文件保存在 {out_dir}/")

if __name__ == '__main__':
    main()
