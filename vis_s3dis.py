import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = './experiments/s3dis_seg_fullft/cfgs/s3dis_seg_fullft_pt'
d = np.load(f'{BASE}/s3dis_area5_pred.npz', allow_pickle=True)
xyz, gt, pred, room = d['xyz'], d['gt'], d['pred'], d['room']

BG = 'black'   # 改 'white' 可一键切换白底
CLASS_NAMES = ['ceiling','floor','wall','beam','column','window','door',
               'table','chair','sofa','bookcase','board','clutter']
if BG == 'black':
    COLORS = np.array([
        [200,200,200],[0,200,0],[0,200,200],[255,255,0],[255,0,255],[255,0,0],
        [0,0,255],[255,128,0],[128,0,255],[0,128,255],[255,0,128],[128,255,0],[128,128,128]
    ]) / 255.0
else:  # 白底需加深浅色类（ceiling 等）
    COLORS = np.array([
        [120,120,120],[0,160,0],[0,150,150],[230,200,0],[200,0,200],[220,0,0],
        [0,0,220],[230,110,0],[110,0,230],[0,110,230],[230,0,110],[110,200,0],[100,100,100]
    ]) / 255.0

rooms, counts = np.unique(room, return_counts=True)
order = np.argsort(counts)[::-1]
cand = [(r, c) for r, c in zip(rooms, counts) if 40 <= c <= 300]
cand.sort(key=lambda t: -t[1])
sel = [r for r, _ in cand[:3]] or list(rooms[order][:3])
print('可视化房间:', sel, '| 底色:', BG)

MAXP = 35000
fig = plt.figure(figsize=(16, 14))
if BG == 'black':
    fig.patch.set_facecolor('black')
text_c = 'white' if BG == 'black' else 'black'
for row, rname in enumerate(sel):
    mask = room == rname
    xyz_r = xyz[mask].reshape(-1, 3)
    gt_r = gt[mask].reshape(-1)
    pred_r = pred[mask].reshape(-1)
    if len(xyz_r) > MAXP:
        idx = np.random.choice(len(xyz_r), MAXP, replace=False)
        xyz_r, gt_r, pred_r = xyz_r[idx], gt_r[idx], pred_r[idx]
    title = rname.replace('_', ' ')
    for col, (lab, t2) in enumerate([(gt_r, 'Ground Truth'), (pred_r, 'HPE Prediction')]):
        ax = fig.add_subplot(len(sel), 2, row * 2 + col + 1, projection='3d')
        if BG == 'black':
            ax.set_facecolor('black')
        ax.scatter(xyz_r[:,0], xyz_r[:,1], xyz_r[:,2], c=COLORS[lab],
                   s=1.2, marker='.', depthshade=False, linewidths=0)
        ax.set_title(f'{title} - {t2}', fontsize=13, color=text_c, pad=2)
        ax.set_axis_off()
        ax.view_init(elev=45, azim=-60)
        try:
            ax.set_box_aspect((np.ptp(xyz_r[:,0]), np.ptp(xyz_r[:,1]), np.ptp(xyz_r[:,2])))
        except Exception:
            pass

handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS[i],
                   markersize=9, label=CLASS_NAMES[i]) for i in range(13)]
fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=9,
           frameon=False, labelcolor=text_c, bbox_to_anchor=(0.5, 0.0),
           columnspacing=2.2, handletextpad=0.4, borderaxespad=0)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f'{BASE}/s3dis_vis.png', dpi=200, bbox_inches='tight')
print('[Saved]', f'{BASE}/s3dis_vis.png')
