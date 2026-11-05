import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import torch, h5py
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from utils.config import get_config
from tools import builder
from pointnet2_ops import pointnet2_utils

def parse_args():
    p = argparse.ArgumentParser()
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
                         log_name='save_pred', distributed=False,
                         local_rank=0, use_gpu=True, num_workers=4,
                         resume=False, experiment_path=f'./experiments/{args.exp_name}/cfgs/{args.exp_name}')
    config = get_config(ns)
    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt)
    model = model.cuda().eval()

    import yaml
    with open('cfgs/dataset_configs/S3DIS.yaml') as f:
        data_root = os.path.expanduser(yaml.safe_load(f)['DATA_PATH'])
    npoints = config.npoints

    with open(os.path.join(data_root, 'room_filelist.txt')) as f:
        room_list = [ln.strip() for ln in f if ln.strip()]
    valid_idx = [i for i, r in enumerate(room_list) if r.startswith('Area_5')]

    offsets, cnt = [], 0
    for fname in sorted(f for f in os.listdir(data_root) if f.endswith('.h5')):
        path = os.path.join(data_root, fname)
        with h5py.File(path, 'r') as h:
            n = len(h['data'])
        offsets.append((path, cnt, cnt + n))
        cnt += n
    assert cnt == len(room_list)
    cache = {}
    def get_block(gidx):
        for path, start, end in offsets:
            if start <= gidx < end:
                if path not in cache:
                    cache[path] = h5py.File(path, 'r')
                h = cache[path]
                return h['data'][gidx - start], h['label'][gidx - start]

    all_xyz, all_gt, all_pred, all_room = [], [], [], []
    BS = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_idx), BS), desc='Infer+save'):
            bg = valid_idx[i:i + BS]
            xyz_ns, lab_ns, xyz_gs = [], [], []
            for gidx in bg:
                d9, lab = get_block(gidx)                    # (4096,9),(4096,)
                a_x, b_x = np.polyfit(d9[:, 0], d9[:, 6], 1) # 恢复全局 x
                a_y, b_y = np.polyfit(d9[:, 1], d9[:, 7], 1) # 恢复全局 y
                x_g = d9[:, 0] + b_x / a_x
                y_g = d9[:, 1] + b_y / a_y
                xyz_g = np.stack([x_g, y_g, d9[:, 2]], 1).astype(np.float32)
                lab_all = lab.astype(np.int64)
                ch = np.random.choice(4096, npoints, replace=False)
                xyz = xyz_g[ch]
                lab_s = lab_all[ch]
                cent = xyz.mean(0)
                xyz_n = xyz - cent
                md = np.sqrt((xyz_n ** 2).sum(1)).max()
                if md > 1e-8:
                    xyz_n = xyz_n / md
                xyz_ns.append(xyz_n); lab_ns.append(lab_s); xyz_gs.append(xyz)
            pts = torch.from_numpy(np.stack(xyz_ns)).cuda()
            fps_idx = pointnet2_utils.furthest_point_sample(pts, npoints).int()
            pts_fps = pointnet2_utils.gather_operation(pts.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            logits = model(pts_fps)
            preds = logits.argmax(dim=1).cpu().numpy()
            fids = fps_idx.cpu().numpy()
            for j, gidx in enumerate(bg):
                fidx = fids[j]
                all_xyz.append(xyz_gs[j][fidx])
                all_gt.append(lab_ns[j][fidx])
                all_pred.append(preds[j])
                all_room.append(room_list[gidx])

    xyz = np.stack(all_xyz); gt = np.stack(all_gt); pred = np.stack(all_pred)
    room = np.array(all_room)
    print('\n[room 样本]', room[:3], 'rooms:', len(set(room)))
    print('[xyz 范围] x:[%.1f,%.1f] y:[%.1f,%.1f] z:[%.1f,%.1f]' % (
        xyz[:,:,0].min(), xyz[:,:,0].max(), xyz[:,:,1].min(), xyz[:,:,1].max(),
        xyz[:,:,2].min(), xyz[:,:,2].max()))
    out = f'./experiments/{args.exp_name}/cfgs/{args.exp_name}/s3dis_area5_pred.npz'
    np.savez(out, xyz=xyz, gt=gt, pred=pred, room=room)
    print('[Saved]', out)

if __name__ == '__main__':
    main()
