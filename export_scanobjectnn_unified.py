import os
import sys
import json
import random
import hashlib
import logging
import argparse

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = "/root/pointmae_peft/Point-MAE"
sys.path = [PROJECT_ROOT] + [
    p for p in sys.path if "hpenext_kd" not in p
]

from utils import parser as project_parser
from utils.logger import get_logger
from utils.config import get_config
from tools import builder
from utils import misc


METHODS = {
    "pretrain": {
        "config": "cfgs/finetune_scan_hardest.yaml",
        "ckpt": "pretrained/pretrain.pth",
        "load_type": "pretrain",
    },
    "lora": {
        "config": "experiments/peft_lora/cfgs/peft_lora/config.yaml",
        "ckpt": "experiments/peft_lora/cfgs/peft_lora/ckpt-best.pth",
        "load_type": "finetuned",
    },
    "hpe": {
        "config": "experiments/peft_hpe/cfgs/peft_hpe/config.yaml",
        "ckpt": "experiments/peft_hpe/cfgs/peft_hpe/ckpt-best.pth",
        "load_type": "finetuned",
    },
    "fullft": {
        "config": "experiments/finetune_scan_hardest_150ep/fair/fair_fullft_scan/config.yaml",
        "ckpt": "experiments/finetune_scan_hardest_150ep/fair/fair_fullft_scan/ckpt-best.pth",
        "load_type": "finetuned",
    },
    "pointlora": {
        "config": "experiments/pointlora_scan/cfgs/pointlora_scan/config.yaml",
        "ckpt": "experiments/pointlora_scan/cfgs/pointlora_scan/ckpt-best.pth",
        "load_type": "finetuned",
    },
    "pointpeft": {
        "config": "experiments/pointpeft_scan/cfgs/pointpeft_scan/config.yaml",
        "ckpt": "experiments/pointpeft_scan/cfgs/pointpeft_scan/ckpt-best.pth",
        "load_type": "finetuned",
    },
}


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sha256_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def parse_own_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        required=True,
        choices=["pretrain", "lora", "hpe", "fullft", "pointlora", "pointpeft"],
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        default="visualization/scanobjectnn_unified",
    )
    return p.parse_args()


def build_project_args(config_path, ckpt_path, method):
    sys.argv = [
        "export_scanobjectnn_unified.py",
        "--config", config_path,
        "--exp_name", f"export_scanobjectnn_{method}",
        "--test",
        "--ckpts", ckpt_path,
        "--launcher", "none",
    ]

    args = project_parser.get_args()
    args.distributed = False
    args.use_gpu = torch.cuda.is_available()
    args.world_size = 1
    args.local_rank = 0
    return args


def main():
    own_args = parse_own_args()
    method = own_args.method
    info = METHODS[method]

    set_deterministic(own_args.seed)

    config_path = os.path.abspath(info["config"])
    ckpt_path = os.path.abspath(info["ckpt"])
    output_dir = os.path.abspath(own_args.output_dir)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置不存在：{config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"权重不存在：{ckpt_path}")

    args = build_project_args(config_path, ckpt_path, method)

    logger = get_logger(f"export_{method}")
    logger.setLevel(logging.ERROR)

    config = get_config(args, logger=logger)

    # 强制使用 hardest 官方测试集。
    dataset_name = config.dataset.test._base_.NAME
    subset = config.dataset.test.others.subset

    if dataset_name != "ScanObjectNN_hardest":
        raise RuntimeError(
            f"数据集错误：期望 ScanObjectNN_hardest，实际为 {dataset_name}"
        )
    if subset != "test":
        raise RuntimeError(
            f"划分错误：期望 test，实际为 {subset}"
        )

    config.dataset.test.others.bs = own_args.batch_size

    _, test_loader = builder.dataset_builder(
        args, config.dataset.test
    )

    dataset_size = len(test_loader.dataset)
    print(f"Method       : {method}")
    print(f"Config       : {config_path}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Dataset      : {dataset_name}")
    print(f"Subset       : {subset}")
    print(f"Dataset size : {dataset_size}")
    print(f"Batch size   : {own_args.batch_size}")

    if dataset_size != 2882:
        raise RuntimeError(
            f"测试样本数错误：期望 2882，实际为 {dataset_size}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = builder.model_builder(config.model)

    if info["load_type"] == "pretrain":
        # 将 Point-MAE 的 MAE_encoder 权重加载到分类骨干中。
        # 分类头保持随机初始化，但这里只使用分类头之前的特征。
        model.load_model_from_ckpt(ckpt_path)
    else:
        # LoRA/HPE 使用各自微调后的完整权重。
        if hasattr(model, '_inject_lora_and_freeze'):
            model._inject_lora_and_freeze()  # PointLoRA: inject before loading ckpt
        builder.load_model(model, ckpt_path, logger=logger)

    model.to(device)
    model.eval()

    captured_features = []

    def feature_hook(module, inputs, output):
        # output: [B, 1 + G, 384]，第一个 token 是 CLS。
        # 仅聚合 patch tokens，避免 Pre-trained 模型中随机初始化的
        # CLS token 直接进入可视化特征。
        tokens = output.detach()
        patch_tokens = tokens[:, 1:, :]
        mean_feature = patch_tokens.mean(dim=1)
        max_feature = patch_tokens.max(dim=1)[0]
        feature = torch.cat(
            [mean_feature, max_feature], dim=-1
        ).cpu()
        captured_features.append(feature)

    # 三种模型统一提取 Transformer 归一化后的 patch tokens，
    # 并采用 mean pooling + max pooling，得到 768 维特征。
    handle = model.norm.register_forward_hook(feature_hook)

    all_labels = []
    all_logits = []
    all_preds = []

    with torch.no_grad():
        for _, _, data in tqdm(
            test_loader,
            desc=f"Exporting {method}",
            total=len(test_loader),
        ):
            points, labels = data
            points = points.float().to(device, non_blocking=True)
            labels = labels.long().reshape(-1)

            # hardest H5 中每个样本本身就是 2048 个点。
            if points.ndim != 3 or points.shape[-1] != 3:
                raise RuntimeError(
                    f"点云形状异常：{tuple(points.shape)}"
                )
            if points.shape[1] != 2048:
                raise RuntimeError(
                    f"点数异常：期望 2048，实际 {points.shape[1]}"
                )

            # 与项目官方测试流程保持一致。
            points = misc.fps(points, config.npoints)
            logits = model(points)

            all_labels.append(labels.cpu())
            all_logits.append(logits.detach().cpu())
            all_preds.append(logits.argmax(dim=-1).detach().cpu())

    handle.remove()

    features = torch.cat(captured_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()

    print("\n导出结果：")
    print("features:", features.shape, features.dtype)
    print("labels  :", labels.shape, labels.dtype)
    print("logits  :", logits.shape, logits.dtype)
    print("preds   :", preds.shape, preds.dtype)

    if features.shape != (2882, 768):
        raise RuntimeError(
            f"特征形状错误：期望 (2882, 768)，实际 {features.shape}"
        )
    if labels.shape != (2882,):
        raise RuntimeError(
            f"标签形状错误：期望 (2882,)，实际 {labels.shape}"
        )
    if logits.shape != (2882, 15):
        raise RuntimeError(
            f"Logits 形状错误：期望 (2882, 15)，实际 {logits.shape}"
        )
    if preds.shape != (2882,):
        raise RuntimeError(
            f"预测形状错误：期望 (2882,)，实际 {preds.shape}"
        )

    for name, array in {
        "features": features,
        "labels": labels,
        "logits": logits,
        "preds": preds,
    }.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} 中存在 NaN 或 Inf")

    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.join(output_dir, method)
    np.save(prefix + "_features.npy", features.astype(np.float32))
    np.save(prefix + "_labels.npy", labels.astype(np.int64))
    np.save(prefix + "_logits.npy", logits.astype(np.float32))
    np.save(prefix + "_preds.npy", preds.astype(np.int64))

    label_hash = sha256_array(labels.astype(np.int64))
    feature_hash = sha256_array(features.astype(np.float32))

    metadata = {
        "method": method,
        "config": config_path,
        "checkpoint": ckpt_path,
        "dataset": dataset_name,
        "subset": subset,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_definition": "concat([mean-pooled patch tokens, max-pooled patch tokens])",
        "npoints": 2048,
        "seed": own_args.seed,
        "label_sha256": label_hash,
        "feature_sha256": feature_hash,
    }

    # 预训练模型分类头未经过训练，准确率无实际含义。
    if method != "pretrain":
        metadata["accuracy"] = float((preds == labels).mean() * 100.0)

    with open(
        prefix + "_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nLabel SHA256 : {label_hash}")
    print(f"Feature SHA256: {feature_hash}")

    if method != "pretrain":
        print(f"Accuracy      : {metadata['accuracy']:.4f}%")
    else:
        print("Accuracy      : 不报告（分类头未微调）")

    print(f"\n已保存至：{output_dir}")


if __name__ == "__main__":
    main()
