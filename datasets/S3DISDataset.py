import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from .build import DATASETS

@DATASETS.register_module()
class S3DIS(Dataset):
    """S3DIS semantic segmentation from PointNet hdf5 blocks.

    room_filelist.txt: one line per block, format 'Area_X_room', block order
    matches the concatenation of ply_data_all_*.h5 files.
    Standard Area-5 protocol: train = Area 1-4,6; test = Area 5.
    Only xyz (first 3 dims) is used, consistent with the pretrained backbone.
    """

    def __init__(self, config):
        super().__init__()
        self.subset = getattr(config, 'subset', 'train')
        self.npoints = getattr(config, 'N_POINTS', 2048)
        self.data_root = getattr(config, 'DATA_PATH', '')
        self._cache = {}

        with open(os.path.join(self.data_root, 'room_filelist.txt'), 'r') as f:
            self.room_list = [ln.strip() for ln in f if ln.strip()]

        # Area-5 split
        if self.subset == 'train':
            self.valid_idx = [i for i, r in enumerate(self.room_list)
                              if not r.startswith('Area_5')]
        else:
            self.valid_idx = [i for i, r in enumerate(self.room_list)
                              if r.startswith('Area_5')]

        # build global block -> (h5 file, within index) offsets
        self.offsets = []  # (path, start, end)
        cnt = 0
        for fname in sorted(f for f in os.listdir(self.data_root) if f.endswith('.h5')):
            path = os.path.join(self.data_root, fname)
            with h5py.File(path, 'r') as h:
                n = len(h['data'])
            self.offsets.append((path, cnt, cnt + n))
            cnt += n
        assert cnt == len(self.room_list), \
            f'block mismatch: h5={cnt} roomlist={len(self.room_list)}'
        print(f'[S3DIS] {self.subset}: {len(self.valid_idx)} blocks '
              f'(Area5 test, total {cnt})')

    def _open(self, path):
        if path not in self._cache:
            self._cache[path] = h5py.File(path, 'r')
        return self._cache[path]

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        gidx = self.valid_idx[idx]
        for path, start, end in self.offsets:
            if start <= gidx < end:
                h = self._open(path)
                pts9 = h['data'][gidx - start].astype(np.float32)   # (4096, 9)
                label = h['label'][gidx - start].astype(np.int64)   # (4096,)
                break

        xyz = pts9[:, :3]
        if len(xyz) >= self.npoints:
            choice = np.random.choice(len(xyz), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(xyz), self.npoints, replace=True)
        xyz, label = xyz[choice], label[choice]

        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
        max_dist = np.sqrt((xyz ** 2).sum(1)).max()
        if max_dist > 1e-8:
            xyz = xyz / max_dist

        return 's3dis', gidx, (torch.from_numpy(xyz), torch.from_numpy(label))
