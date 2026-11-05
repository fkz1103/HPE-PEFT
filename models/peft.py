import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============== LoRA ==============
class LoRALayer(nn.Module):
    """Low-Rank Adaptation for Linear layers"""
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        std = 0.02
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        # x: [B, N, in_dim]  →  x @ lora_A @ lora_B * scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAQKV(nn.Module):
    """LoRA for Attention QKV projection"""
    def __init__(self, qkv: nn.Linear, dim, rank=4, alpha=1.0):
        super().__init__()
        self.qkv = qkv  # frozen pretrained
        self.lora = LoRALayer(dim, dim * 3, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.qkv(x) + self.lora(x)


# ============== Adapter ==============
class Adapter(nn.Module):
    """Bottleneck adapter: dim -> bottleneck -> dim"""
    def __init__(self, dim, bottleneck=64, act_layer=nn.GELU):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = act_layer()
        self.up = nn.Linear(bottleneck, dim)
        # Zero-init the up projection for stable training
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


# ============== Prompt Tuning ==============
class PromptTuning(nn.Module):
    """Learnable prompt tokens prepended to the sequence"""
    def __init__(self, num_prompts=16, dim=384):
        super().__init__()
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, x):
        # x: [B, N, D]
        prompts = self.prompts.expand(x.size(0), -1, -1)
        return torch.cat([prompts, x], dim=1)

    @property
    def num_prompts(self):
        return self.prompts.size(1)


# ============== Utility: Freeze backbone ==============
def freeze_backbone(model, exclude_keywords=['cls_head_finetune', 'prompt', 'lora', 'adapter']):
    """Freeze all parameters except those containing exclude_keywords"""
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in exclude_keywords):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    total = frozen_params + trainable_params
    print(f"[PEFT] Frozen: {frozen_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M "
          f"({100*trainable_params/total:.2f}%)")
    return trainable_params, frozen_params


# ============== Count trainable params ==============
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============== SSF (Scaling & Shifting Your Features) ==============
class SSF(nn.Module):
    """y = gamma * x + beta"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.gamma + self.beta


class SSFLayer(nn.Module):
    """Wrap a frozen Linear with SSF scale & shift, interface identical to Linear"""
    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.layer = layer          # frozen pretrained
        self.ssf = SSF(layer.out_features)

    def forward(self, x):
        return self.ssf(self.layer(x))
