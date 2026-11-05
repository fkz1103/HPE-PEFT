import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import parser, misc
from utils.config import cfg_from_yaml_file
from tools import builder

def main():
    args = parser.get_args()
    # env flags normally set in main.py
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.world_size = 1
    if not hasattr(args, 'local_rank'):
        args.local_rank = 0

    config = cfg_from_yaml_file(args.config)
    config.dataset.test.others.bs = config.total_bs

    # test loader
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    print(f'[Eval] test samples: {len(test_dataloader.dataset)}')

    # model (PEFT applied automatically via config)
    base_model = builder.model_builder(config.model)
    if hasattr(base_model, '_inject_lora_and_freeze'):
        base_model._inject_lora_and_freeze()  # PointLoRA: inject before loading ckpt
    builder.load_model(base_model, args.ckpts, logger=None)
    base_model = base_model.eval().cuda()

    npoints = config.npoints
    preds, labels, taxonomies = [], [], []
    with torch.no_grad():
        for taxonomy_ids, model_ids, data in test_dataloader:
            points = data[0].cuda()
            label = data[1].view(-1).cpu()
            points = misc.fps(points, npoints)
            logits = base_model(points)
            pred = logits.argmax(-1).view(-1).cpu()
            preds.append(pred)
            labels.append(label)
            taxonomies.extend(list(taxonomy_ids))

    pred = torch.cat(preds).numpy()
    label = torch.cat(labels).numpy()
    C = config.model.cls_dim

    acc = (pred == label).mean() * 100
    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (label, pred), 1)
    np.save(os.path.join(args.experiment_path, 'confusion_matrix.npy'), cm)

    ds = test_dataloader.dataset
    ds_classes = list(getattr(ds, 'classes', []) or [])
    class_names = ds_classes if len(ds_classes) == C else ['cls_%02d' % c for c in range(C)]
    per_cls = np.array([(pred[label == c] == c).sum() / max(1, (label == c).sum()) * 100 for c in range(C)])
    mAcc = per_cls.mean()

    print('=' * 66)
    print(f'Method: {config.model.get("peft_method", "full")}  ckpt: {args.ckpts}')
    print(f'Overall Acc = {acc:.4f}%   mAcc = {mAcc:.4f}%')
    print('-' * 66)
    for c in range(C):
        print(f'  {class_names[c]:<22s} acc = {per_cls[c]:6.2f}%  (n={int((label==c).sum())})')
    print('=' * 66)

    with open(os.path.join(args.experiment_path, 'per_class_acc.txt'), 'w') as f:
        f.write(f'Overall {acc:.4f}  mAcc {mAcc:.4f}\n')
        for c in range(C):
            f.write(f'{class_names[c]} {per_cls[c]:.4f}\n')
    print(f'[Eval] saved confusion_matrix.npy & per_class_acc.txt -> {args.experiment_path}')

if __name__ == '__main__':
    main()
