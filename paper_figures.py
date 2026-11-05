import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os, sys

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

MODELNET_CLASSES = [
    'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
    'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
    'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
    'person','piano','plant','radio','range_hood','sink','sofa','stairs',
    'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox'
]

def load_data(dataset):
    parts = dataset.split('_')[0]
    base = f'visualization/{dataset}'
    return {
        'features': np.load(f'{base}/{parts}_features.npy'),
        'logits': np.load(f'{base}/{parts}_logits.npy'),
        'labels': np.load(f'{base}/{parts}_labels.npy'),
        'preds': np.load(f'{base}/{parts}_preds.npy'),
    }

def get_class_names(n_classes, dataset_name):
    if 'modelnet' in dataset_name.lower() and n_classes == 40:
        return MODELNET_CLASSES
    return [f'C{i}' for i in range(n_classes)]

def fig1_tsne(data, dname, path):
    features, labels, acc = data['features'], data['labels'], data['acc']
    np.random.seed(42)
    n_per_class = 60
    sidx = []
    for c in np.unique(labels):
        ic = np.where(labels == c)[0]
        if len(ic) > n_per_class:
            ic = np.random.choice(ic, n_per_class, replace=False)
        sidx.extend(ic.tolist())
    sidx = np.array(sidx)
    print(f"  t-SNE on {len(sidx)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sidx)-1),
                max_iter=500, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(features[sidx])
    labels_s = labels[sidx]
    n_classes = len(np.unique(labels))
    cnames = get_class_names(n_classes, dname)
    cmap1, cmap2 = plt.cm.tab20, plt.cm.tab20b
    colors = [cmap1(i/20) if i < 20 else cmap2((i-20)/20) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in np.unique(labels_s):
        mask = labels_s == c
        cn = cnames[int(c)] if int(c) < len(cnames) else f'C{int(c)}'
        ax.scatter(emb[mask,0], emb[mask,1], c=[colors[int(c)]], s=15, alpha=0.7, label=cn, edgecolors='none')
    ax.set_title(f't-SNE of {dname} Features (Acc: {acc:.2f}%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xticks([]); ax.set_yticks([])
    h, l = ax.get_legend_handles_labels()
    if n_classes > 15:
        ax.legend(h[:15], l[:15], loc='upper right', ncol=2, fontsize=6, markerscale=0.5)
    else:
        ax.legend(loc='best', ncol=2, fontsize=7, markerscale=0.8)
    fig.savefig(path, dpi=300); plt.close()
    print(f"  -> {path}")

def fig2_per_class(data, dname, path):
    labels, preds = data['labels'], data['preds']
    n_classes = len(np.unique(labels))
    cnames = get_class_names(n_classes, dname)
    pc = []
    for c in range(n_classes):
        m = labels == c
        if m.sum() > 0:
            pc.append((c, (preds[m]==labels[m]).mean()*100, m.sum()))
    pc.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, max(5, n_classes*0.25)))
    names = [cnames[c[0]] if c[0]<len(cnames) else f'C{c[0]}' for c in pc]
    accs = [c[1] for c in pc]
    colors = ['#d62728' if a<70 else ('#ff7f0e' if a<85 else '#2ca02c') for a in accs]
    bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(x=np.mean(accs), color='gray', linestyle='--', linewidth=1, label=f'Mean: {np.mean(accs):.1f}%')
    ax.set_xlabel('Accuracy (%)'); ax.set_title(f'Per-Class Accuracy on {dname}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105); ax.legend()
    for bar, v in zip(bars, accs):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=5)
    fig.savefig(path, dpi=300); plt.close()
    print(f"  -> {path}")

def fig3_topk(data, dname, path):
    logits, labels = data['logits'], data['labels']
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ks = [1, 3, 5]
    accs = [np.any(np.argsort(-probs, axis=1)[:,:k]==labels.reshape(-1,1),axis=1).mean()*100 for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([f'Top-{k}' for k in ks], accs, color=['#1f77b4','#ff7f0e','#2ca02c'], width=0.5, edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.2f}%', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105); ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{dname} Top-k Accuracy', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(path, dpi=300); plt.close()
    print(f"  -> {path}")

def fig4_confusion(data, dname, path):
    labels, preds = data['labels'], data['preds']
    n_classes = len(np.unique(labels))
    cnames = get_class_names(n_classes, dname)
    cm = confusion_matrix(labels, preds)
    errs = cm.sum(axis=1) - np.diag(cm)
    top10 = np.argsort(-errs)[:10]
    cm_sub = cm[np.ix_(top10, top10)]
    names_sub = [cnames[c] if c<len(cnames) else f'C{c}' for c in top10]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_sub, cmap='Blues', aspect='auto')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm_sub[i,j]), ha='center', va='center', fontsize=7,
                    color='white' if cm_sub[i,j]>cm_sub.max()*0.7 else 'black')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(names_sub, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names_sub, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{dname} - Top-10 Confusing Classes', fontsize=13, fontweight='bold')
    fig.colorbar(im, shrink=0.8)
    fig.savefig(path, dpi=300); plt.close()
    print(f"  -> {path}")

def fig5_confidence(data, dname, path):
    logits, labels, preds = data['logits'], data['labels'], data['preds']
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    conf = probs[np.arange(len(labels)), preds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf[labels==preds], bins=50, alpha=0.7, label=f'Correct ({len(conf[labels==preds])})', color='#2ca02c', edgecolor='white', linewidth=0.5)
    ax.hist(conf[labels!=preds], bins=50, alpha=0.7, label=f'Wrong ({len(conf[labels!=preds])})', color='#d62728', edgecolor='white', linewidth=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Confidence Score'); ax.set_ylabel('Number of Samples')
    ax.set_title(f'{dname} - Confidence Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(path, dpi=300); plt.close()
    print(f"  -> {path}")

def fig6_pca(data, dname, path):
    features, labels = data['features'], data['labels']
    pca = PCA(n_components=2)
    feat_pca = pca.fit_transform(features)
    n_classes = len(np.unique(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab20 if n_classes<=20 else plt.cm.nipy_spectral
    sc = ax1.scatter(feat_pca[:,0], feat_pca[:,1], c=labels, cmap=cmap, s=3, alpha=0.5)
    ax1.set_title(f'{dname} PCA', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(sc, ax=ax1, shrink=0.8)
    pca_f = PCA(n_components=min(20, features.shape[1]))
    pca_f.fit(features)
    cum = np.cumsum(pca_f.explained_variance_ratio_)*100
    ax2.plot(range(1,len(cum)+1), cum, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% variance')
    ax2.set_xlabel('Number of Components'); ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title(f'{dname} - PCA Explained Variance', fontsize=12, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close()
    print(f"  -> {path}")

def generate_all(dataset_name):
    data = load_data(dataset_name)
    acc = (data['labels'] == data['preds']).mean() * 100
    data['acc'] = acc
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name} | Acc: {acc:.2f}%")
    print(f"{'='*50}")
    out_dir = f'visualization/{dataset_name}/paper_figures'
    os.makedirs(out_dir, exist_ok=True)
    print("\n[1/6] t-SNE...");     fig1_tsne(data, dataset_name, f'{out_dir}/fig1_tsne.png')
    print("[2/6] Per-Class Acc..."); fig2_per_class(data, dataset_name, f'{out_dir}/fig2_per_class_acc.png')
    print("[3/6] Top-k...");         fig3_topk(data, dataset_name, f'{out_dir}/fig3_topk.png')
    print("[4/6] Confusion Top-10..."); fig4_confusion(data, dataset_name, f'{out_dir}/fig4_confusion_top10.png')
    print("[5/6] Confidence...");    fig5_confidence(data, dataset_name, f'{out_dir}/fig5_confidence.png')
    print("[6/6] PCA...");          fig6_pca(data, dataset_name, f'{out_dir}/fig6_pca.png')
    print(f"\nAll saved to {out_dir}/")

if __name__ == '__main__':
    generate_all(sys.argv[1])
