import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchvision import transforms

# 修复路径
sys.path = [p for p in sys.path if 'hpenext' not in p]
sys.path.insert(0, '.')

from tools import builder
from utils.config import get_config
import argparse

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ShapeNetPart 部件颜色映射
PART_CMAP = plt.cm.tab20
PART_CMAP2 = plt.cm.tab20b

def get_part_color(part_id):
    if part_id < 20:
        return PART_CMAP(part_id / 20)
    else:
        return PART_CMAP2((part_id - 20) / 20)

def plot_3d_pointcloud(points, labels, title, ax, elev=30, azim=120):
    colors = [get_part_color(int(l)) for l in labels]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=2, alpha=0.9, edgecolors='none')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor('white')
    max_r = max(abs(points).max(), 0.5)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)

def main():
    # 直接构造 args
    class Args:
        config = 'cfgs/shapenet_part_seg.yaml'
        ckpts = 'experiments/shapenet_part_seg/cfgs/peft_hpe_seg_perlayer/ckpt-best.pth'
        exp_name = 'vis_seg_hpe'
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
        ckpts = 'experiments/shapenet_part_seg/cfgs/peft_hpe_seg_perlayer/ckpt-best.pth'

    args = Args()
    args.experiment_path = './experiments/vis_seg'
    args.tfboard_path = './experiments/vis_seg/tfboard'
    
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

    # 选择要可视化的类别
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
    all_preds, all_gts = [], []
    npoints = config.npoints

    print("=== 推理测试集 ===")
    from pointnet2_ops import pointnet2_utils

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

        all_preds.append(pred.cpu())
        all_gts.append(seg_label.cpu())

        # 收集各类别样本
        tax_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]
        for cat_name, cat_id in vis_categories.items():
            if tax_id == cat_id and len(samples_by_category[cat_name]) < 2:
                samples_by_category[cat_name].append({
                    'points': points[0].cpu().numpy(),
                    'pred': pred[0].cpu().numpy(),
                    'gt': seg_label[0].cpu().numpy(),
                    'model_id': str(model_ids[0]),
                })

        if idx % 50 == 0:
            print(f"  Processed {idx} samples...")

        # 收集够了就提前停
        total_needed = sum(2 for _ in vis_categories)
        collected = sum(len(v) for v in samples_by_category.values())
        if collected >= total_needed and idx > 100:
            print("  (collected enough samples, stopping early)")
            break

    # 计算 mIoU
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    from tools.runner_seg import compute_part_miou
    miou = compute_part_miou(all_preds.flatten(), all_gts.flatten(), num_parts=50)
    acc = (all_preds == all_gts).sum().float() / all_gts.numel() * 100
    print(f"\n=== 测试结果 ===")
    print(f"mIoU = {miou:.4f}")
    print(f"Acc  = {acc:.2f}%")

    # 生成可视化
    out_dir = 'visualization/shapenet_part_seg_hpe'
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== 生成可视化 → {out_dir}/ ===")

    for cat_name, samples in samples_by_category.items():
        for i, sample in enumerate(samples):
            pts = sample['points']
            pred_labels = sample['pred']
            gt_labels = sample['gt']
            model_id = sample['model_id']

            for elev, azim, vname in [(30, 120, 'side'), (30, 30, 'front'), (15, 210, 'top')]:
                fig = plt.figure(figsize=(18, 6))
                
                ax1 = fig.add_subplot(131, projection='3d')
                plot_3d_pointcloud(pts, gt_labels, f'{cat_name} - GT', ax1, elev=elev, azim=azim)

                ax2 = fig.add_subplot(132, projection='3d')
                plot_3d_pointcloud(pts, pred_labels, f'{cat_name} - Pred', ax2, elev=elev, azim=azim)

                ax3 = fig.add_subplot(133, projection='3d')
                wrong = pred_labels != gt_labels
                wrong_pct = wrong.mean() * 100
                colors = []
                for p, g in zip(pred_labels, gt_labels):
                    if p == g:
                        colors.append(get_part_color(int(p)))
                    else:
                        colors.append((1.0, 0.2, 0.2, 1.0))
                ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=colors, s=2, alpha=0.9, edgecolors='none')
                ax3.set_title(f'Errors ({wrong_pct:.1f}%)', fontsize=12,
                             fontweight='bold', pad=10)
                ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
                ax3.view_init(elev=elev, azim=azim)
                ax3.set_facecolor('white')
                mr = max(abs(pts).max(), 0.5)
                ax3.set_xlim(-mr, mr); ax3.set_ylim(-mr, mr); ax3.set_zlim(-mr, mr)

                plt.tight_layout(pad=2)
                fname = f'{out_dir}/{cat_name}_{model_id}_{vname}.png'
                fig.savefig(fname, dpi=300)
                plt.close()
                print(f"  ✓ {fname}")

    # 论文风格三面板
    print("\n=== 生成论文风格三面板图 ===")
    for cat_name, samples in samples_by_category.items():
        if not samples:
            continue
        sample = samples[0]
        pts = sample['points']
        pred_labels = sample['pred']
        gt_labels = sample['gt']

        fig = plt.figure(figsize=(20, 7))
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d_pointcloud(pts, gt_labels, 'Ground Truth', ax1, elev=25, azim=100)
        ax2 = fig.add_subplot(132, projection='3d')
        plot_3d_pointcloud(pts, pred_labels, 'HPE-PEFT Prediction', ax2, elev=25, azim=100)
        ax3 = fig.add_subplot(133, projection='3d')
        wrong = pred_labels != gt_labels
        colors = [get_part_color(int(p)) if p == g else (1.0, 0.2, 0.2, 1.0) for p, g in zip(pred_labels, gt_labels)]
        ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=2, alpha=0.9, edgecolors='none')
        wrong_r = wrong.mean() * 100
        ax3.set_title(f'Errors ({wrong_r:.1f}%)', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
        ax3.view_init(elev=25, azim=100); ax3.set_facecolor('white')
        mr = max(abs(pts).max(), 0.5)
        ax3.set_xlim(-mr, mr); ax3.set_ylim(-mr, mr); ax3.set_zlim(-mr, mr)

        plt.suptitle(f'{cat_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = f'{out_dir}/{cat_name}_3panel.png'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {fname}")

    print(f"\n✅ 完成！所有可视化保存至 {out_dir}/")

if __name__ == '__main__':
    main()
