import numpy as np

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    gt   = np.asarray(d['gt']).ravel().astype(np.int64)
    pred = np.asarray(d['pred']).ravel().astype(np.int64)
    room = np.asarray(d['room'])
    return gt, pred, room

def room_miou(gt, pred, n_cls=13):
    ious = []
    for c in range(n_cls):
        inter = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        ious.append(inter / union if union > 0 else float('nan'))
    return np.nanmean(ious)

gt_h, pred_h, room_h = load_npz('./experiments/s3dis_seg_hpe/cfgs/s3dis_seg_hpe_pt/s3dis_area5_pred.npz')
gt_f, pred_f, room_f = load_npz('./experiments/s3dis_seg_fullft/cfgs/s3dis_seg_fullft_pt/s3dis_area5_pred.npz')

rep_h = gt_h.shape[0] // room_h.shape[0]   # 2048
rep_f = gt_f.shape[0] // room_f.shape[0]
print('points per block:', rep_h, rep_f)

ROOMS = ['Area_5_hallway_5', 'Area_5_office_40', 'Area_5_conferenceRoom_2']
for room in ROOMS:
    m_h = np.repeat(room_h == room, rep_h)
    m_f = np.repeat(room_f == room, rep_f)
    miou_h = room_miou(gt_h[m_h], pred_h[m_h])
    miou_f = room_miou(gt_f[m_f], pred_f[m_f])
    counts = np.bincount(gt_h[m_h], minlength=13)
    frac = counts / counts.sum()
    top = ', '.join(f'{i}({frac[i]*100:.0f}%)' for i in np.argsort(frac)[::-1][:5])
    print(f'[{room}] points(HPE)={int(m_h.sum())}')
    print(f'  HPE    room-mIoU = {miou_h*100:.2f}%')
    print(f'  FullFT room-mIoU = {miou_f*100:.2f}%')
    print(f'  top classes (id,占比): {top}')
