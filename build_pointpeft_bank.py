import os
import sys
import argparse

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path = [PROJECT_ROOT] + [p for p in sys.path if "hpenext_kd" not in p]

from utils import parser as project_parser
from utils.logger import get_logger
from utils.config import get_config
from tools import builder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="训练用 config（决定数据集与模型结构，须与 pointpeft 训练 config 同源）")
    p.add_argument("--ckpts", default="./pretrained/pretrain.pth")
    p.add_argument("--out", required=True, help="bank 保存路径，如 ./ckpts/bank_pointpeft_modelnet40.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--exp_name", default="build_pointpeft_bank")
    own = p.parse_args()

    sys.argv = [
        "build_pointpeft_bank.py",
        "--config", own.config,
        "--exp_name", own.exp_name,
        "--ckpts", own.ckpts,
        "--launcher", "none",
    ]
    args = project_parser.get_args()
    args.distributed = False
    args.use_gpu = torch.cuda.is_available()
    args.world_size = 1
    args.local_rank = 0

    logger = get_logger(f"build_bank_{own.exp_name}")
    config = get_config(args, logger=logger)

    config.dataset.train.others.bs = own.batch_size
    _, train_loader = builder.dataset_builder(args, config.dataset.train)

    model = builder.model_builder(config.model)
    model.load_model_from_ckpt(own.ckpts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    feats = []
    with torch.no_grad():
        for _, _, data in train_loader:
            points, _ = data
            points = points.float().to(device, non_blocking=True)
            if points.ndim != 3 or points.shape[-1] != 3:
                raise RuntimeError(f"点云形状异常：{tuple(points.shape)}")
            f = model.extract_bank_feature(points)  # (B, D)
            feats.append(f.cpu())

    X = torch.cat(feats, dim=0)
    X = F.normalize(X.float(), dim=-1)

    out_dir = os.path.dirname(os.path.abspath(own.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(X, own.out)

    print(f"训练集样本数 : {X.shape[0]}")
    print(f"特征维度     : {X.shape[1]}")
    print(f"bank 已保存  : {own.out}")


if __name__ == "__main__":
    main()
