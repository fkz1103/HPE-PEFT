from types import SimpleNamespace
from models.segmentation import PointTransformerSeg

cfg = SimpleNamespace(
    trans_dim=384,
    depth=12,
    drop_path_rate=0.1,
    num_heads=6,
    group_size=32,
    num_group=512,
    encoder_dims=384,
    peft_method='hpe',
    peft_rank=4,
    peft_bottleneck=64,
    freeze_backbone=True,
    peft_per_layer=True,
    dice_weight=0.2,
)

model = PointTransformerSeg(cfg)

groups = {
    "基础 HPE": "hpe_module",
    "逐层缩放": "layer_hpe_scale",
    "逐层偏置": "layer_hpe_bias",
    "跳跃融合": "skip_fusion",
    "特征传播": "fp",
    "解码器位置编码": "decoder_pos_embed",
    "分割头": "seg_head",
}

print("=" * 80)

counted = 0
for title, keyword in groups.items():
    n = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and keyword in name
    )
    counted += n
    print(f"{title:20s}: {n:10,d} ({n / 1e6:.6f}M)")

trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
total = sum(p.numel() for p in model.parameters())

print("-" * 80)
print(f"{'已分类参数':20s}: {counted:10,d}")
print(f"{'总可训练参数':20s}: {trainable:10,d} ({trainable / 1e6:.6f}M)")
print(f"{'模型总参数':20s}: {total:10,d} ({total / 1e6:.6f}M)")
print(f"{'可训练参数比例':20s}: {100 * trainable / total:.4f}%")
print(f"{'未分类参数':20s}: {trainable - counted:10,d}")
print("=" * 80)
