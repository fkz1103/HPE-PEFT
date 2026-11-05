import torch, numpy as np, os, sys
sys.path.append('/root/pointmae_peft/Point-MAE/extensions/chamfer_dist')
sys.path = ['/root/pointmae_peft/Point-MAE'] + [p for p in sys.path if 'hpenext_kd' not in p]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

def run():
    config_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    exp_name = sys.argv[3]
    dataset_name = sys.argv[4] if len(sys.argv) > 4 else exp_name
    
    from utils import parser
    from utils.logger import get_logger
    from utils.config import get_config
    from tools import builder
    import logging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 用 parser.get_args() 获取完整参数
    sys.argv = ['vis_cls.py', '--config', config_path, '--exp_name', exp_name,
                '--test', '--ckpts', ckpt_path, '--launcher', 'none']
    args = parser.get_args()
    
    # 补充 builder 需要的属性（parser.get_args 不会设置这些）
    args.distributed = False
    args.use_gpu = torch.cuda.is_available()
    args.world_size = 1
    args.sync_bn = False
    
    logger = get_logger('vis_logger')
    logger.setLevel(logging.ERROR)
    config = get_config(args, logger=logger)
    
    # 数据集
    config.dataset.test.others.bs = 1
    print("Loading dataset...")
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    # 模型
    print("Loading model...")
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    base_model.to(device)
    base_model = torch.nn.DataParallel(base_model)
    base_model.eval()
    
    npoints = config.npoints
    out_dir = f'visualization/{exp_name}'
    os.makedirs(out_dir, exist_ok=True)
    
    # Hook 提取 concat_f 特征
    hook_features = []
    def hook_fn(m, i, o):
        hook_features.append(i[0].detach().cpu())
    handle = base_model.module.cls_head_finetune[0].register_forward_hook(hook_fn)
    
    all_logits, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            points = data[0].to(device)
            label = data[1].to(device)
            points = misc.fps(points, npoints)
            logits = base_model(points)
            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())
            all_preds.append(logits.argmax(-1).cpu())
    
    handle.remove()
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    features = torch.cat(hook_features, dim=0).numpy() if hook_features else all_logits
    
    cls_dim = config.model.cls_dim
    class_names = [f'C{i}' for i in range(cls_dim)]
    names_file = 'data/ModelNet/modelnet40_normal_resampled/modelnet40_shape_names.txt'
    if os.path.exists(names_file):
        with open(names_file) as f:
            names = [l.strip() for l in f.readlines()]
        if len(names) == cls_dim:
            class_names = names
    
    acc = (all_labels == all_preds).mean() * 100
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Test Acc: {acc:.2f}% | Samples: {len(all_labels)}")
    print(f"{'='*50}\n")
    
    # 1. t-SNE
    print("t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_labels)-1), max_iter=1000)
    emb = tsne.fit_transform(features)
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(emb[:,0], emb[:,1], c=all_labels, cmap='tab20', s=8, alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)) if len(class_names) <= 20 else None)
    plt.title(f'{dataset_name} t-SNE (Acc: {acc:.2f}%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/tsne.png', dpi=150); plt.close()
    print(f"  -> {out_dir}/tsne.png")
    
    # 2. Confusion Matrix
    print("Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    present = sorted(set(all_labels) | set(all_preds))
    present_names = [class_names[i] for i in present]
    cm_f = cm[np.ix_(present, present)]
    plt.figure(figsize=(16, 14) if len(present)>20 else (10, 8))
    sns.heatmap(cm_f, annot=len(present)<=20, fmt='d', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names)
    plt.title(f'{dataset_name} Confusion Matrix (Acc: {acc:.2f}%)', fontsize=14)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/confusion_matrix.png', dpi=150); plt.close()
    print(f"  -> {out_dir}/confusion_matrix.png")
    
    # 3. Top-k
    print("Top-k Accuracy...")
    probs = torch.from_numpy(all_logits).softmax(dim=-1)
    labels_t = torch.from_numpy(all_labels)
    k_vals = [1, 3, 5]
    topk_acc = []
    for k in k_vals:
        topk = probs.topk(k, dim=-1).indices
        acc_k = (topk == labels_t.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100
        topk_acc.append(acc_k)
    plt.figure(figsize=(6, 4))
    bars = plt.bar([f'Top-{k}' for k in k_vals], topk_acc, color='steelblue')
    for i, v in enumerate(topk_acc):
        plt.text(i, v+0.5, f'{v:.2f}%', ha='center', fontsize=11)
    plt.title(f'{dataset_name} Top-k Accuracy'); plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/topk_accuracy.png', dpi=150); plt.close()
    print(f"  -> {out_dir}/topk_accuracy.png")
    
    print(f"\nAll saved to {out_dir}/")
    
    # 查看文件大小确认生成成功
    for f in os.listdir(out_dir):
        sz = os.path.getsize(os.path.join(out_dir, f)) // 1024
        print(f"  {f}: {sz}KB")

if __name__ == '__main__':
    from utils import misc
    run()
