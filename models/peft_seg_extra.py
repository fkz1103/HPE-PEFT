import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """PointLoRA 用的通用 LoRA 包装(qkv / proj 任意 nn.Linear)。

    关键设计: 直接引用原 Linear 的 weight/bias(Parameter 对象),
    参数名路径保持不变(attn.qkv.weight), 预训练 ckpt 按原 key 照常加载,
    不会出现 wrap 后 key 对不上导致骨干权重加载失败的问题。
    """
    def __init__(self, base: nn.Linear, rank=8, alpha=8.0):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scaling = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)          # 初始零输出,等价于原模型

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) \
               + self.scaling * self.lora_up(self.lora_down(x))
