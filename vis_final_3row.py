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
    'font.size': 12,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.edgecolor': 'none',
})

PART_CMAP = plt.cm.tab20
PART_CMAP2 = plt.cm.tab20b

def get_part_color(part_id):
    if part_id < 20:
        return PART_CMAP(part_id / 20)
    else:
        return PART_CMAP2((part_id - 20) / 20)

def plot_clean(ax, pts, labels, elev=30, azim=120):
    """干净的无边框3D点云图"""
    colors = [get_part_color(int(l)) for l in labels]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=6, alpha=0.9, edgecolors='none', linewidth=0)
    
    # 完全去掉所有边框、网格、刻度
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')
    
    # 隐藏所有边框线
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_visible(False)
        axis.pane.set_visible(False)
        axis.pane.set_edgecolor('none')
        axis.pane.set_alpha(0)
    
    # 隐藏frame
    ax._axis3don = False
    
    ax.view_init(elev=elev, azim=azim)
    mr = max(abs(pts).max(), 0.5)
    ax.set_xlim(-mr, mr); ax.set_ylim(-mr, mr); ax.set_zlim(-mr, mr)
    ax.set_position([ax.get_position().x0, ax.get_position().y0,
                     ax.get_position().width * 1.1, ax.get_position().height * 1.1])

def main():
    # ===== 加载两个模型 =====
    class Args:
        launcher = 'none'; local_rank = 0; test = True
        use_gpu = torch.cuda.is_available(); distributed = False
        num_workers = 4; seed = 0; deterministic = False; sync_bn = False
        experiment_path = './experiments/vis_final'; tfboard_path = './experiments/vis_final/tfboard'
        log_name = 'vis_final'; val_freq = 1; vote = False; resume = False
        finetune_model = False; scratch_model = False; mode = None
        way = -1; shot = -1; fold = -1; start_ckpts = None; loss = 'cd1'

    # 模型1: HPE-PEFT
    args1 = Args()
    args1.config = 'cfgs/shapenet_part_seg.yaml'
    args1.ckpts = 'experiments/shapenet_part_seg/cfgs/peft_hpe_seg_perlayer/ckpt-best.pth'
    config1 = get_config(args1, logger=None); config1.dataset.test.others.bs = 1
    model_hpe = builder.model_builder(config1.model)
    builder.load_model(model_hpe, args1.ckpts, logger=None)
    model_hpe.cuda(); model_hpe.eval()

    # 模型2: HPENet v2 (全量微调)
    args2 = Args()
    args2.config = 'cfgs/shapenet_part_seg_ft.yaml'
    args2.ckpts = 'experiments/shapenet_part_seg_ft/cfgs/shapenet_part_seg_ft/ckpt-best.pth'
    config2 = get_config(args2, logger=None); config2.dataset.test.others.bs = 1
    model_ft = builder.model_builder(config2.model)
    builder.load_model(model_ft, args2.ckpts, logger=None)
    model_ft.cuda(); model_ft.eval()

    # 加载数据
    _, test_dataloader = builder.dataset_builder(args1, config1.dataset.test)

    vis_categories = {
        'Airplane': '02691156', 'Chair': '03001627', 'Guitar': '03467517',
        'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512',
        'Pistol': '03948459', 'Table': '04379243',
    }

    samples = {name: None for name in vis_categories}
    npoints = config1.npoints
    from pointnet2_ops import pointnet2_utils
    torch.set_grad_enabled(False)

    print("=== 推理（两个模型同时跑）===")
    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        points = data[0].cuda(); seg_label = data[1].cuda()
        fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        seg_label = torch.gather(seg_label, 1, fps_idx.long())

        pred_hpe = model_hpe(points).argmax(dim=1)
        pred_ft = model_ft(points).argmax(dim=1)

        tax_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]
        for cat_name, cat_id in vis_categories.items():
            if tax_id == cat_id and samples[cat_name] is None:
                samples[cat_name] = {
                    'points': points[0].cpu().numpy(),
                    'gt': seg_label[0].cpu().numpy(),
                    'hpe': pred_hpe[0].cpu().numpy(),
                    'ft': pred_ft[0].cpu().numpy(),
                }

        if idx % 50 == 0: print(f"  Processed {idx}...")
        if all(v is not None for v in samples.values()) and idx > 100:
            print("  (所有类别样本收集完毕)"); break

    # ===== 生成最终 3行×3列 对比图 =====
    out_dir = 'visualization/shapenet_part_seg_hpe/paper_figures'
    os.makedirs(out_dir, exist_ok=True)

    views = [
        (30, 120, 'Side View'),
        (30, 30,  'Front View'),
        (15, 210, 'Top View'),
    ]

    # 行标签（每个方法一行）
    row_configs = [
        ('Ground Truth', 'gt'),
        ('HPENet v2 (Full FT)', 'ft'),
        ('HPE-PEFT (Ours)', 'hpe'),
    ]

    print("\n=== 生成最终三行对比图 ===")
    for cat_name, sample in samples.items():
        if sample is None: continue
        pts = sample['points']

        fig = plt.figure(figsize=(20, 16))

        for row_idx, (method_name, key) in enumerate(row_configs):
            labels = sample[key]
            for col_idx, (elev, azim, view_name) in enumerate(views):
                # 手动创建3D子图，不使用subplot以精确控制间距
                left = 0.01 + col_idx * 0.33
                bottom = 0.01 + (2 - row_idx) * 0.33
                width = 0.32
                height = 0.32
                
                ax = fig.add_axes([left, bottom, width, height], projection='3d')
                
                if key == 'gt':
                    colors = [get_part_color(int(l)) for l in labels]
                else:
                    colors = [get_part_color(int(l)) for l in labels]
                
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                          c=colors, s=8, alpha=0.9, edgecolors='none', linewidth=0)
                
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.grid(False)
                ax.set_facecolor('white')
                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis.line.set_visible(False)
                    axis.pane.set_visible(False)
                    axis.pane.set_edgecolor('none')
                    axis.pane.set_alpha(0)
                ax._axis3don = False
                
                ax.view_init(elev=elev, azim=azim)
                mr = max(abs(pts).max(), 0.5)
                ax.set_xlim(-mr, mr); ax.set_ylim(-mr, mr); ax.set_zlim(-mr, mr)

                # === 列标题（只最上面一行显示）===
                if row_idx == 0:
                    ax.set_title(view_name, fontsize=16, fontweight='bold',
                                pad=8, y=1.08)

                # === 行标题（只最左边一列显示）===
                if col_idx == 0:
                    ax.text2D(-0.15, 0.5, method_name,
                             transform=ax.transAxes,
                             fontsize=15, fontweight='bold',
                             ha='center', va='center',
                             rotation='vertical')

        # 最上面的大标题：类别名
        fig.suptitle(f'{cat_name}', fontsize=20, fontweight='bold',
                    x=0.5, y=0.98)

        plt.tight_layout(pad=0.5)
        fname = f'{out_dir}/{cat_name}_3row_comparison.png'
        fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  ✓ {fname}")

    print(f"\n✅ 完成！文件保存在 {out_dir}/ 下 *_3row_comparison.png")

if __name__ == '__main__':
    main()
