import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names = ['ceiling','floor','wall','beam','column','window','door',
         'table','chair','sofa','bookcase','board','clutter']
files = [
    ('experiments/s3dis_seg_hpe/cfgs/s3dis_seg_hpe_pt/s3dis_area5_eval.npz', 'HPE-PEFT', 's3dis_conf_hpe.png'),
    ('experiments/s3dis_seg_fullft/cfgs/s3dis_seg_fullft_pt/s3dis_area5_eval.npz', 'FullFT', 's3dis_conf_fullft.png'),
]
for path, title, out in files:
    d = np.load(path)
    conf = d['conf'].astype(int)
    conf_n = conf / (conf.sum(axis=1, keepdims=True) + 1e-9)   # 行归一化
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(conf_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(13)); ax.set_yticks(range(13))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
    ax.set_title(f'S3DIS Area 5 Confusion Matrix ({title})')
    for i in range(13):
        for j in range(13):
            v = conf_n[i, j]
            if v > 0.5:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white')
            elif v > 0.03:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, color='black')
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print('saved', out)
