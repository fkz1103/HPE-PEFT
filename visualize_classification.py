import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os, sys, argparse
sys.path.append('/root/pointmae_peft/Point-MAE/extensions/chamfer_dist')
from tqdm import tqdm

sys.path = [p for p in sys.path if 'hpenext_kd' not in p]

def extract_features_and_predict(base_model, dataloader, npoints, device):
    """提取特征和预测结果"""
    base_model.eval()
    
    all_features = []
    all_logits = []
    all_labels = []
    all_preds = []
    
    # 注册 hook 提取倒数第二层特征 (concat_f)
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())
    
    handle = base_model.module.cls_head_finetune[0].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataloader, desc="Extracting features")):
            points = data[0].to(device)
            label = data[1].to(device)
            
            # FPS downsampling
            from utils import misc
            points = misc.fps(points, npoints)
            
            _ = base_model(points)
            
            all_labels.append(label.cpu())
    
    handle.remove()
    
    all_labels = torch.cat(all_labels, dim=0)
    
    # 重新过一遍提取 logits
    base_model.eval()
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(dataloader, desc="Extracting logits")):
            points = data[0].to(device)
            from utils import misc
            points = misc.fps(points, npoints)
            
            logits = base_model(points)
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(-1).cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    # 整理 features (按 batch 拼接)
    if len(features) > 0:
        all_features = torch.cat(features, dim=0)
    else:
        all_features = all_logits
    
    return all_features.numpy(), all_logits.numpy(), all_labels.numpy(), all_preds.numpy()


def plot_tsne(features, labels, class_names, save_path, title="t-SNE Visualization"):
    """t-SNE 可视化"""
    print(f"Running t-SNE on {features.shape} features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, 
                         cmap='tab20', s=8, alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)) if len(class_names) <= 20 else None)
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE dim 1', fontsize=12)
    plt.ylabel('t-SNE dim 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE to {save_path}")


def plot_confusion_matrix(labels, preds, class_names, save_path, title="Confusion Matrix"):
    """混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(16, 14) if len(class_names) > 20 else (10, 8))
    
    # 只显示有样本的类别
    present_classes = sorted(set(labels) | set(preds))
    present_names = [class_names[i] for i in present_classes]
    cm_filtered = cm[np.ix_(present_classes, present_classes)]
    
    acc = np.mean(labels == preds) * 100
    sns.heatmap(cm_filtered, annot=len(present_classes) <= 20, fmt='d', 
                cmap='Blues', xticklabels=present_names, yticklabels=present_names)
    plt.title(f'{title} (Acc: {acc:.2f}%)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_feature_distribution(features, labels, class_names, save_path, title="Feature Distribution"):
    """特征分布直方图（PCA 降维后的前2维）"""
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    feat_pca = pca.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 一维分布
    axes[0].hist(feat_pca[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].set_title(f'{title} - PC1 Distribution', fontsize=12)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('Frequency')
    
    # 二维散点
    axes[1].scatter(feat_pca[:, 0], feat_pca[:, 1], c=labels, 
                   cmap='tab20', s=5, alpha=0.6)
    axes[1].set_title(f'{title} - PCA Visualization', fontsize=12)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature distribution to {save_path}")


def plot_top_k_accuracy(logits, labels, save_path, k_values=[1, 3, 5]):
    """Top-k 准确率曲线"""
    topk_acc = []
    probs = torch.from_numpy(logits).softmax(dim=-1)
    labels_t = torch.from_numpy(labels)
    
    for k in k_values:
        topk = probs.topk(k, dim=-1).indices
        acc = (topk == labels_t.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100
        topk_acc.append(acc)
    
    plt.figure(figsize=(6, 4))
    plt.bar([f'Top-{k}' for k in k_values], topk_acc, color='steelblue')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-k Accuracy')
    for i, v in enumerate(topk_acc):
        plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top-k accuracy to {save_path}")


def visualize_classification(config_path, ckpt_path, exp_name, cls_dim, class_names, dataset_type='modelnet'):
    """主函数"""
    from utils import parser, misc
    from utils.logger import get_logger
    from utils.config import get_config
    from tools import builder
    import logging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 解析参数
    args = parser.get_args()
    args.config = config_path
    args.ckpts = ckpt_path
    args.exp_name = exp_name
    args.test = True
    args.launcher = 'none'
    
    # 抑制日志
    logger = get_logger('vis_logger')
    logger.setLevel(logging.ERROR)
    
    config = get_config(args, logger=logger)
    
    # 加载数据集
    print("Loading dataset...")
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    # 加载模型
    print("Loading model...")
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    base_model.to(device)
    base_model = torch.nn.DataParallel(base_model)
    
    npoints = config.npoints
    
    # 提取特征
    print("Extracting features...")
    features, logits, labels, preds = extract_features_and_predict(
        base_model, test_dataloader, npoints, device
    )
    
    # 创建输出目录
    out_dir = f'visualization/{exp_name}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 总体精度
    acc = np.mean(labels == preds) * 100
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_type}")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Total samples: {len(labels)}")
    print(f"{'='*50}\n")
    
    # 1. t-SNE
    plot_tsne(features, labels, class_names, 
              f'{out_dir}/tsne.png', 
              title=f'{dataset_type} - t-SNE (Acc: {acc:.2f}%)')
    
    # 2. 混淆矩阵
    plot_confusion_matrix(labels, preds, class_names,
                         f'{out_dir}/confusion_matrix.png',
                         title=f'{dataset_type}')
    
    # 3. 特征分布
    plot_feature_distribution(features, labels, class_names,
                             f'{out_dir}/feature_distribution.png',
                             title=f'{dataset_type}')
    
    # 4. Top-k 准确率
    plot_top_k_accuracy(logits, labels, f'{out_dir}/topk_accuracy.png')
    
    print(f"\nAll visualizations saved to {out_dir}/")
    return acc


if __name__ == '__main__':
    # ModelNet40
    modelnet_classes = [f'Class_{i}' for i in range(40)]
    
    # 读取实际的 ModelNet40 类别名称
    modelnet_names_path = 'data/ModelNet/modelnet40_normal_resampled/modelnet40_shape_names.txt'
    if os.path.exists(modelnet_names_path):
        with open(modelnet_names_path, 'r') as f:
            modelnet_names = [line.strip() for line in f.readlines()]
        if len(modelnet_names) == 40:
            modelnet_classes = modelnet_names
    
    visualize_classification(
        config_path='cfgs/peft_hpe_modelnet40.yaml',
        ckpt_path='experiments/peft_hpe_modelnet40/cfgs/peft_hpe_modelnet40/ckpt-best.pth',
        exp_name='modelnet40_hpe',
        cls_dim=40,
        class_names=modelnet_classes,
        dataset_type='ModelNet40'
    )
