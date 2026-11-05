import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from utils.config import get_config
from tools import builder
from pointnet2_ops import pointnet2_utils

CLASS_NAMES = ['ceiling','floor','wall','beam','column','window',
               'door','table','chair','sofa','bookcase','board','clutter']

def parse_args():
    p = argparse.ArgumentParser(description='S3DIS Area5 scene-level evaluation')
    p.add_argument('--config', type=str, default='cfgs/s3dis_seg_hpe.yaml')
    p.add_argument('--exp_name', type=str, default='s3dis_seg_hpe')
    p.add_argument('--ckpt', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    ckpt = args.ckpt or f'./experiments/{args.exp_name}/cfgs/{args.exp_name}/ckpt-last.pth'
    if not os.path.exists(ckpt):
        ckpt = f'./experiments/{args.exp_name}/cfgs/{args.exp_name}/ckpt-best.pth'

    ns = SimpleNamespace(config=args.config, exp_name=args.exp_name, ckpts=ckpt,
                         log_name='eval_s3dis', distributed=False,
                         local_rank=0, use_gpu=True, num_workers=4, resume=False, experiment_path=f'./experiments/{args.exp_name}/cfgs/{args.exp_name}')
    config = get_config(ns)

    _, test_loader = builder.dataset_builder(ns, config.dataset.test)
    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt)
    model = model.cuda().eval()
    print(f'[Eval] blocks: {len(test_loader.dataset)}, ckpt: {ckpt}')

    npoints = config.npoints
    all_pred, all_gt = [], []
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_loader, desc='Testing Area5')):
            points = data[0].cuda()
            seg_label = data[1].cuda()
            fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            seg_label = torch.gather(seg_label, 1, fps_idx.long())
            logits = model(points)          # (B, C, N)
            all_pred.append(logits.argmax(dim=1).cpu())
            all_gt.append(seg_label.cpu())

    all_pred = torch.cat(all_pred, dim=0).flatten().numpy()
    all_gt = torch.cat(all_gt, dim=0).flatten().numpy()

    conf = np.zeros((13, 13), dtype=np.int64)
    np.add.at(conf, (all_pred, all_gt), 1)
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(1) - tp
    fn = conf.sum(0) - tp
    iou = tp / (tp + fp + fn + 1e-6)
    per_acc = [tp[i]/(tp[i]+fn[i]) if tp[i]+fn[i] > 0 else 0.0 for i in range(13)]
    miou, oa, macc = iou.mean(), tp.sum()/conf.sum(), np.mean(per_acc)

    print('\n======== S3DIS Area5 (scene-level) ========')
    print(f'{"class":<10}{"IoU":>8}{"Acc":>8}')
    for i, n in enumerate(CLASS_NAMES):
        print(f'{n:<10}{iou[i]*100:>7.2f}%{per_acc[i]*100:>7.2f}%')
    print('-' * 30)
    print(f'mIoU  : {miou*100:.2f}%')
    print(f'OA    : {oa*100:.2f}%')
    print(f'mAcc  : {macc*100:.2f}%')
    print('\nConfusion matrix (rows=pred, cols=gt):')
    print(conf)
    np.savez(f'./experiments/{args.exp_name}/cfgs/{args.exp_name}/s3dis_area5_eval.npz',
             conf=conf, iou=iou, per_acc=np.array(per_acc), miou=miou, oa=oa, macc=macc)
    print('\n[Saved] s3dis_area5_eval.npz')

if __name__ == '__main__':
    main()
