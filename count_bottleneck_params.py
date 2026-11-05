from types import SimpleNamespace
from models.Point_MAE import PointTransformer

dims = [16, 32, 64, 128]

print("=" * 100)
print(f"{'b':>5} {'HPE':>12} {'分类头':>12} {'其他':>10} {'可训练参数':>14} {'参数量(M)':>12} {'占总模型(%)':>12}")
print("-" * 100)

for b in dims:
    cfg = SimpleNamespace(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        cls_dim=15,
        num_heads=6,
        group_size=32,
        num_group=128,
        encoder_dims=384,
        peft_method='hpe',
        peft_bottleneck=b,
        freeze_backbone=True,
    )

    model = PointTransformer(cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    hpe = sum(
        p.numel() for p in model.hpe_module.parameters()
        if p.requires_grad
    )
    head = sum(
        p.numel() for p in model.cls_head_finetune.parameters()
        if p.requires_grad
    )
    other = trainable - hpe - head
    ratio = 100.0 * trainable / total

    print(
        f"{b:5d} "
        f"{hpe:12,d} "
        f"{head:12,d} "
        f"{other:10,d} "
        f"{trainable:14,d} "
        f"{trainable / 1e6:12.6f} "
        f"{ratio:12.4f}"
    )

print("=" * 100)
