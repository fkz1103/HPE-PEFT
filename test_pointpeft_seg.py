import os
import sys
import subprocess

import torch

CFG = "cfgs/shapenet_part_seg_pointpeft.yaml"
CKPT = "pretrained/pretrain.pth"
BANK = "./pretrained/bank_shapenetpart_train.pt"


def main():
    if not torch.cuda.is_available():
        sys.exit("[smoke] x 未检测到 GPU(KNN CUDA 组件需要 GPU),请在 GPU 空闲时运行")
    if not os.path.exists(BANK):
        print("[smoke] point-prior bank 缺失,自动构建 ShapeNetPart bank(约 1-3 分钟)...")
        r = subprocess.run([sys.executable, "build_pointpeft_bank.py",
                            "--config", CFG, "--ckpts", CKPT, "--out", BANK,
                            "--batch_size", "32"])
        if r.returncode != 0:
            sys.exit("[smoke] x bank 构建失败,详见上方输出")

    import yaml
    from easydict import EasyDict
    from models import build_model_from_cfg

    with open(CFG) as f:
        cfg = EasyDict(yaml.safe_load(f))
    model = build_model_from_cfg(cfg.model).cuda()
    model.load_model_from_ckpt(CKPT)
    if model.bank is None or model.bank.numel() == 0:
        sys.exit("[smoke] x bank 未加载,检查 cfg 中 bank_path")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] Trainable = {n_train / 1e3:.1f}K / {n_total / 1e3:.1f}K "
          f"({100.0 * n_train / n_total:.2f}%)")

    model.train()
    pts = torch.rand(2, 2048, 3).cuda()
    logits = model(pts)
    assert tuple(logits.shape) == (2, 50, 2048), f"输出形状异常: {tuple(logits.shape)}"
    target = torch.randint(0, 50, (2, 2048)).cuda()
    loss, acc = model.get_loss_acc(logits, target)
    loss.backward()
    assert model.prompt_tokens.grad is not None, "prompt_tokens 没有梯度"
    assert model.blocks.blocks[0].attn.qkv.weight.grad is None, "骨干权重不应有梯度"
    print(f"[smoke] forward/backward OK, loss = {loss.item():.4f}")
    print("[smoke] V 全部通过,流水线可无缝进入 Point-PEFT 训练")


if __name__ == "__main__":
    main()
