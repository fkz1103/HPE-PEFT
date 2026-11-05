import sys, os
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 3:
    print("usage: python vis_s3dis_parts_white.py <hpe_pred.npz> <fullft_pred.npz>")
    sys.exit(1)

def pick(d, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError("none of %s in %s" % (names, list(d.keys())))

def load_all(d):
    xyz = pick(d, "xyz", "coords", "coord")
    gt  = pick(d, "gt", "label", "labels")
    pr  = pick(d, "pred", "preds", "pred_label")
    rm  = pick(d, "room", "rooms")
    if xyz.ndim == 3:                    # (N, P, 3) block-major
        N, P, _ = xyz.shape
        xyz = xyz.reshape(-1, 3)
    else:                                # (N*P, 3) flat
        P = xyz.shape[0] // rm.shape[0]
    return xyz, np.asarray(gt).reshape(-1), np.asarray(pr).reshape(-1), rm, P

h = np.load(sys.argv[1])
f = np.load(sys.argv[2])
print("HPE keys:", list(h.keys()))
print("FT  keys:", list(f.keys()))

xyz_h, gt_h, pred_h, room_h, NPT = load_all(h)
xyz_f, gt_f, pred_f, room_f, NPT2 = load_all(f)
assert NPT == NPT2 == 2048, "expect 2048 points per block, got %d/%d" % (NPT, NPT2)
print("HPE xyz", xyz_h.shape, "room", room_h.shape)
print("FT  xyz", xyz_f.shape, "room", room_f.shape)
print("points per block:", NPT)

ROOMS = [("hallway_5", "Area_5_hallway_5"), ("office_40", "Area_5_office_40"), ("conferenceRoom_2", "Area_5_conferenceRoom_2")]
ELEV, AZIM, MAXP, S = 45, -60, 35000, 1.2
CMAP = {0:(.55,.55,.55),1:(.90,.20,.30),2:(.20,.60,.90),3:(.90,.70,.20),
        4:(.40,.25,.80),5:(.95,.40,.10),6:(.30,.85,.40),7:(.85,.55,.20),
        8:(.10,.50,.75),9:(.80,.80,.20),10:(.60,.30,.60),11:(.95,.85,.30),
        12:(.35,.65,.85)}
OUT = "./s3dis_vis_parts_white"
os.makedirs(OUT, exist_ok=True)

def sample_idx(n, maxp=MAXP):
    if n <= maxp:
        return None
    return np.random.default_rng(0).choice(n, maxp, replace=False)

def render(fname, xyz, lab, idx):
    if idx is not None:
        xyz, lab = xyz[idx], lab[idx]
    # [白底版] 原脚本三处 black 全部改 white
    fig = plt.figure(figsize=(6, 6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    c = np.array([CMAP[int(l)] for l in lab])
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=S, marker=".")
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0)
    plt.close(fig)

for tag5, room in ROOMS:
    mh = np.repeat(room_h == room, NPT)
    mf = np.repeat(room_f == room, NPT)
    xyzh, gth, prh = xyz_h[mh], gt_h[mh], pred_h[mh]
    xyzf, prf      = xyz_f[mf], pred_f[mf]
    ih, iff = sample_idx(len(xyzh)), sample_idx(len(xyzf))
    print("%s: HPE %d blocks | FT %d blocks" % (room, len(xyzh)//NPT, len(xyzf)//NPT))
    render(f"{OUT}/{tag5}_gt.png",     xyzh, gth, ih)
    render(f"{OUT}/{tag5}_fullft.png", xyzf, prf, iff)
    render(f"{OUT}/{tag5}_hpe.png",    xyzh, prh, ih)
    print("[Saved] " + tag5)

print("Done. 9 white-background figures in", OUT)
