import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from knn_cuda import KNN
from timm.models.layers import trunc_normal_
from .build import MODELS
from utils import misc
from utils.logger import *
from .peft import freeze_backbone
from .hpe_peft import HPE_PEFT


class FeaturePropagation(nn.Module):
    """Upsample point features from group centers back to original points."""

    def __init__(self, in_channels, mlp_channels, k=3):
        super().__init__()
        self.k = k
        mlp = []
        for i in range(len(mlp_channels)):
            ch = mlp_channels[i]
            mlp.append(nn.Conv1d(in_channels if i == 0 else mlp_channels[i-1], ch, 1))
            mlp.append(nn.BatchNorm1d(ch))
            mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x_center, center, xyz):
        B, G, C = x_center.shape
        _, N, _ = xyz.shape

        dist = torch.cdist(xyz, center)
        dist_topk, idx_topk = torch.topk(dist, self.k, dim=2, largest=False)

        weight = 1.0 / (dist_topk + 1e-8)
        weight = weight / weight.sum(dim=2, keepdim=True)

        idx_topk = idx_topk.unsqueeze(-1).expand(-1, -1, -1, C)
        x_center_expand = x_center.unsqueeze(1).expand(-1, N, -1, -1)
        neighbor_features = torch.gather(x_center_expand, 2, idx_topk)

        weight = weight.unsqueeze(-1)
        interpolated = (neighbor_features * weight).sum(dim=2)

        x = interpolated.permute(0, 2, 1)
        x = self.mlp(x)
        return x


@MODELS.register_module()
class PointTransformerSeg(nn.Module):
    """Point-MAE for ShapeNetPart part segmentation with skip connections.

    Enhanced with per-layer HPE-PEFT for improved segmentation quality.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads
        self.num_part = getattr(config, "num_classes", 50)

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.peft_method = getattr(config, 'peft_method', None)
        self.peft_rank = getattr(config, 'peft_rank', 4)
        self.peft_alpha = getattr(config, 'peft_alpha', 1.0)
        self.peft_bottleneck = getattr(config, 'peft_bottleneck', 64)
        self.peft_num_prompts = getattr(config, 'peft_num_prompts', 16)
        self.freeze_backbone_peft = getattr(config, 'freeze_backbone', True) if self.peft_method else False

        # === [PointLoRA 新增] 多尺度 token 选择 / LoRA 超参 ===
        self.peft_plora_rank = int(getattr(config, 'peft_plora_rank', 8))
        self.peft_plora_alpha = float(getattr(config, 'peft_plora_alpha', 8.0))
        self.peft_plora_k1 = int(getattr(config, 'peft_plora_k1', 32))
        self.peft_plora_k2 = int(getattr(config, 'peft_plora_k2', 8))

        # === [PointPEFT 新增] Point-PEFT 超参(与分类 port 同名同默认) ===
        self.peft_prompt_layers = int(getattr(config, 'peft_prompt_layers', 6))
        self.peft_prompt_len = int(getattr(config, 'peft_prompt_len', 10))
        self.peft_adapter_bottleneck = int(getattr(config, 'peft_adapter_bottleneck', 16))
        self.ga_n_center = int(getattr(config, 'peft_ga_n_center', 32))
        self.ga_knn = int(getattr(config, 'peft_ga_knn', 16))
        self.bank_path = getattr(config, 'bank_path', None)

        # === [优化] 逐层 HPE 调制开关 ===
        self.peft_per_layer = getattr(config, 'peft_per_layer', True)
        # === [优化] Dice Loss 权重 ===
        self.dice_weight = getattr(config, 'dice_weight', 0.2)

        from .Point_MAE import Group, Encoder, TransformerEncoder
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        use_adapter = (self.peft_method == 'adapter')
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            use_adapter=use_adapter,
            adapter_bottleneck=self.peft_bottleneck,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # HPE-PEFT
        if self.peft_method == 'hpe':
            self.hpe_module = HPE_PEFT(
                group_size=self.group_size,
                trans_dim=self.trans_dim,
                bottleneck=self.peft_bottleneck,
            )
            # === [优化] 逐层 HPE 调制参数 ===
            if self.peft_per_layer:
                self.layer_hpe_scale = nn.Parameter(torch.ones(self.depth, self.trans_dim))
                self.layer_hpe_bias = nn.Parameter(torch.zeros(self.depth, self.trans_dim))
                print_log(f'[Seg] Using per-layer HPE modulation (depth={self.depth})', logger='Transformer')
            else:
                self.layer_hpe_scale = None
                self.layer_hpe_bias = None
                print_log(f'[Seg] Using shared HPE (no per-layer modulation)', logger='Transformer')

            print_log(f'[Seg] Using HPE-PEFT with bottleneck={self.peft_bottleneck}', logger='Transformer')
        else:
            self.hpe_module = None
            self.layer_hpe_scale = None
            self.layer_hpe_bias = None

        # === Prompt Tuning ===
        if self.peft_method == 'prompt':
            from .peft import PromptTuning
            self.prompt_tuning = PromptTuning(num_prompts=self.peft_num_prompts, dim=self.trans_dim)
            print_log(f'[Seg] Using PromptTuning (num_prompts={self.peft_num_prompts})', logger='Transformer')
        else:
            self.prompt_tuning = None

        # === [PointLoRA 新增] 多尺度 token 选择模块(与分类 port 同构) ===
        if self.peft_method == 'pointlora':
            from .point_lora_model import MaskPredictor
            self.lora_mask_predictor = MaskPredictor(token_dim=self.trans_dim)
            self.center_divider1 = Group(num_group=128, group_size=32)
            self.center_divider2 = Group(num_group=64, group_size=64)
            print_log(f'[Seg] Using PointLoRA (rank={self.peft_plora_rank}, '
                      f'alpha={self.peft_plora_alpha}, K1={self.peft_plora_k1}, '
                      f'K2={self.peft_plora_k2})', logger='Transformer')
        else:
            self.lora_mask_predictor = None

        # === [PointPEFT 新增] group prompts + point-prior 检索 + 逐层 GA-Adapter ===
        if self.peft_method == 'pointpeft':
            from .point_peft_model import GAAdapter, GroupAttention
            self.prompt_tokens = nn.Parameter(
                torch.zeros(self.peft_prompt_layers, self.peft_prompt_len, self.trans_dim))
            self.prompt_coords = nn.Parameter(torch.zeros(self.peft_prompt_len, 3))
            self.prior_mlp = nn.Sequential(
                nn.Linear(self.trans_dim, self.peft_adapter_bottleneck),
                nn.GELU(),
                nn.Linear(self.peft_adapter_bottleneck, self.trans_dim),
            )
            self.group_attn = GroupAttention(dim=self.trans_dim, num_heads=self.num_heads)
            self.ga_adapters = nn.ModuleList([
                GAAdapter(dim=self.trans_dim, bottleneck=self.peft_adapter_bottleneck)
                for _ in range(self.depth)
            ])
            # 冻结骨干独立副本(point-prior bank 特征提取,键路径与 ckpt 对齐)
            self.bank_encoder = Encoder(encoder_channel=self.encoder_dims)
            self.bank_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.trans_dim),
            )
            self.bank_blocks = TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.num_heads,
                use_adapter=False,
            )
            self.bank_norm = nn.LayerNorm(self.trans_dim)
            self.register_buffer('bank', torch.zeros(0), persistent=False)
            self.knn_geo = KNN(k=self.ga_knn, transpose_mode=True)
            print_log(f'[Seg] Using Point-PEFT (prompt_layers={self.peft_prompt_layers}, '
                      f'prompt_len={self.peft_prompt_len}, bottleneck={self.peft_adapter_bottleneck}, '
                      f'ga_n_center={self.ga_n_center}, ga_knn={self.ga_knn})', logger='Transformer')

        # === Skip Connection ===
        self.skip_fusion = nn.Sequential(
            nn.Linear(self.trans_dim * 2, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )

        # Feature Propagation decoder
        self.fp = FeaturePropagation(
            in_channels=self.trans_dim,
            mlp_channels=[self.trans_dim, self.trans_dim],
            k=3,
        )

        # === [优化] 解码器 HPE 注入（可选，提高边界预测能力）===
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, self.trans_dim),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, self.num_part, 1),
        )

        self.loss_ce = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

        # === [PointPEFT 新增] prompt 裸 Parameter 初始化(apply 不作用于 Parameter) ===
        if self.peft_method == 'pointpeft':
            trunc_normal_(self.prompt_tokens, std=.02)
            trunc_normal_(self.prompt_coords, std=.02)

        # Apply PEFT
        if self.peft_method == 'lora':
            self._apply_lora()
        # === [PointLoRA 新增] ===
        if self.peft_method == 'pointlora':
            self._apply_pointlora()
        if self.freeze_backbone_peft:
            self._freeze_for_peft()

    def _apply_lora(self):
        from .peft import LoRAQKV
        from .Point_MAE import Block
        for name, module in self.named_modules():
            if isinstance(module, Block):
                attn = module.attn
                dim = attn.qkv.in_features
                attn.qkv = LoRAQKV(attn.qkv, dim, rank=self.peft_rank, alpha=self.peft_alpha)
        print_log(f'[Seg] Applied LoRA (rank={self.peft_rank}, alpha={self.peft_alpha})', logger='Transformer')

    # === [PointLoRA 新增] QKV + Proj 双注入(与分类 port 一致) ===
    def _apply_pointlora(self):
        from .peft_seg_extra import LoRALinear
        for block in self.blocks.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv,
                                        rank=self.peft_plora_rank, alpha=self.peft_plora_alpha)
            block.attn.proj = LoRALinear(block.attn.proj,
                                        rank=self.peft_plora_rank, alpha=self.peft_plora_alpha)
        print_log(f'[Seg] PointLoRA injected on qkv+proj '
                  f'(rank={self.peft_plora_rank}, alpha={self.peft_plora_alpha})', logger='Transformer')

    def _freeze_for_peft(self):
        exclude_keywords = ['seg_head', 'fp', 'skip_fusion', 'decoder_pos_embed']
        if self.peft_method == 'lora':
            exclude_keywords.append('lora')
        elif self.peft_method == 'adapter':
            exclude_keywords.append('adapter')
        elif self.peft_method == 'prompt':
            exclude_keywords.append('prompt_tuning')
        elif self.peft_method == 'hpe':
            exclude_keywords.append('hpe')
        # === [PointLoRA 新增] ===
        elif self.peft_method == 'pointlora':
            exclude_keywords.append('lora')
            exclude_keywords.append('mask_predictor')
        # === [PointPEFT 新增] 冻结口径与分类 port 一致 ===
        elif self.peft_method == 'pointpeft':
            exclude_keywords.append('prompt')
            exclude_keywords.append('prior_mlp')
            exclude_keywords.append('ga_adapters')
            exclude_keywords.append('group_attn')
            exclude_keywords.append('bias')
        trainable, frozen = freeze_backbone(self, exclude_keywords=exclude_keywords)
        print_log(f'[Seg] PEFT={self.peft_method}, Trainable={trainable/1e3:.1f}K', logger='Transformer')
        # === [PointPEFT 新增] bank 提取器整体强制冻结(防止 bias 关键字解冻) ===
        if self.peft_method == 'pointpeft':
            for n, p in self.named_parameters():
                if n.startswith('bank_'):
                    p.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu')

            if 'base_model' in ckpt:
                state_dict = ckpt['base_model']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            mapped = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '', 1)
                if new_k.startswith('MAE_encoder.'):
                    new_k = new_k[len('MAE_encoder.'):]
                if new_k.startswith('MAE_decoder') or new_k.startswith('decoder_pos_embed') \
                   or new_k.startswith('increase_dim') or new_k == 'mask_token':
                    continue
                mapped[new_k] = v

            model_keys = set(self.state_dict().keys())
            loadable = {k: v for k, v in mapped.items() if k in model_keys}

            print_log(f'[Seg] Loading pretrain: {len(loadable)}/{len(mapped)} keys matched', logger='Transformer')

            msg = self.load_state_dict(loadable, strict=False)

            # === [PointPEFT 新增] 冻结骨干副本加载 + point-prior bank 文件 ===
            if self.peft_method == 'pointpeft':
                bank_sd = {}
                for k, v in mapped.items():
                    if k.startswith('blocks.'):
                        bank_sd['bank_blocks.' + k] = v
                    elif k.startswith('encoder.'):
                        bank_sd['bank_encoder.' + k] = v
                    elif k.startswith('pos_embed.'):
                        bank_sd['bank_pos_embed.' + k] = v
                    elif k.startswith('norm.'):
                        bank_sd['bank_norm.' + k] = v
                if bank_sd:
                    self.load_state_dict(bank_sd, strict=False)
                if self.bank is None or self.bank.numel() == 0:
                    self._load_bank()

            missing = [m for m in msg.missing_keys if not m.startswith('seg_head')
                      and not m.startswith('fp.') and not m.startswith('hpe')
                      and not m.startswith('skip_fusion.')
                      and not m.startswith('decoder_pos_embed.')
                      and m != 'module.cls_token' and m != 'module.cls_pos'
                      and m != 'cls_token' and m != 'cls_pos']
            # === [PointPEFT 新增] 缺失键过滤(新注入模块本就不在 ckpt 中) ===
            if self.peft_method == 'pointpeft':
                missing = [m for m in missing
                           if not m.startswith(('prompt_', 'prior_mlp', 'group_attn',
                                                'ga_adapters', 'bank_'))]
            if missing:
                print_log(f'[Seg] Missing (expected): {missing[:5]}...', logger='Transformer')
            print_log(f'[Seg] Loaded checkpoint from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!', logger='Transformer')

    # === [PointPEFT 新增] point-prior bank 与几何索引(与分类 port 逐字同构) ===
    def _load_bank(self):
        path = self.bank_path
        if path and os.path.exists(path):
            bank = torch.load(path, map_location='cpu')
            bank = F.normalize(bank.float(), dim=-1)
            # [PointPEFT 修复] 与模型当前设备对齐(兼容 .cuda() 之前/之后调用)
            device = next(self.parameters()).device
            bank = bank.to(device)
            self.register_buffer('bank', bank, persistent=False)
            print_log(f'[PointPEFT] Loaded point-prior bank from {path}, shape {tuple(bank.shape)}',
                      logger='Transformer')
        else:
            print_log(f'[PointPEFT] WARNING: point-prior bank not found at {path}. '
                      f'Run test_pointpeft_seg.py / build_pointpeft_bank.py first.', logger='Transformer')

    def _set_bank_eval(self):
        self.bank_encoder.eval()
        self.bank_blocks.eval()
        self.bank_norm.eval()

    def _bank_feature_from_group(self, neighborhood, center):
        """冻结预训练骨干前向:patch tokens mean pooling -> (B, D)"""
        self._set_bank_eval()
        with torch.no_grad():
            t = self.bank_encoder(neighborhood)
            p = self.bank_pos_embed(center)
            x = t + p
            for blk in self.bank_blocks.blocks:
                x = blk(x)
            x = self.bank_norm(x)
            return x.mean(dim=1)

    def extract_bank_feature(self, pts):
        """外部(bank 生成脚本)使用的特征提取接口"""
        neighborhood, center = self.group_divider(pts)
        return self._bank_feature_from_group(neighborhood, center)

    def _gen_prompt_prior(self, ft):
        """cosine 检索 top-(K-2) -> softmax 聚合 -> concat -> bottleneck MLP"""
        K = self.peft_prompt_len
        q = F.normalize(ft, dim=-1)
        S = q @ self.bank.t()
        topv, topi = torch.topk(S, K - 2, dim=-1)
        Xk = self.bank[topi]
        w = F.softmax(topv, dim=-1).unsqueeze(-1)
        FA = (w * Xk).sum(dim=1, keepdim=True)
        inp = torch.cat([q.unsqueeze(1), FA, Xk], dim=1)
        return self.prior_mlp(inp)

    def _fps_index(self, xyz, npoint):
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), device=device)
        ar = torch.arange(B, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[ar, farthest].unsqueeze(1)
            dist = ((xyz - centroid) ** 2).sum(-1)
            distance = torch.minimum(distance, dist)
            farthest = torch.max(distance, dim=-1)[1]
        return centroids

    def _geo_index(self, coords):
        """返回 (cen_idx, nn_idx, flat_idx, flat_w, inv),全部无梯度"""
        with torch.no_grad():
            B, Nt, _ = coords.shape
            cen_idx = self._fps_index(coords, self.ga_n_center)
            ar = torch.arange(B, device=coords.device)
            coords_c = coords[ar[:, None], cen_idx]
            _, nn_idx = self.knn_geo(coords, coords_c)
            coords_n = coords[ar[:, None, None], nn_idx]
            dist2 = ((coords_n - coords_c[:, :, None, :]) ** 2).sum(-1)
            w = 1.0 / (dist2 + 1e-8)
            flat_idx = nn_idx.reshape(B, -1)
            flat_w = w.reshape(B, -1)
            w_sum = torch.zeros(B, Nt, device=coords.device).scatter_add_(1, flat_idx, flat_w)
            inv = 1.0 / (w_sum + 1e-8)
            return cen_idx, nn_idx, flat_idx, flat_w, inv

    def _dice_loss(self, pred, target, smooth=1.0):
        B, C, N = pred.shape
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=C)
        target_one_hot = target_one_hot.permute(0, 2, 1).float()

        intersection = (pred_softmax * target_one_hot).sum(dim=2)
        union = pred_softmax.sum(dim=2) + target_one_hot.sum(dim=2)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1.0 - dice.mean()
        return loss

    def forward(self, pts):
        """
        Args:
            pts: [B, N, 3] input point cloud
        Returns:
            seg_logits: [B, num_part, N] per-point part logits
        """
        xyz = pts  # [B, N, 3]

        # Group and encode
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # [B, G, C]

        pos_base = self.pos_embed(center)  # [B, G, D] 基础位置编码

        if self.hpe_module is not None:
            hpe_out = self.hpe_module(neighborhood, center)  # [B, G, D] HPE 输出
            if self.peft_per_layer and self.layer_hpe_scale is not None:
                pass  # 在下面的循环中逐层处理
            else:
                pos = pos_base + hpe_out
        else:
            hpe_out = None
            pos = pos_base

        # === [PointLoRA 新增] 多尺度 token 选择(与分类 port 同一逻辑) ===
        if self.peft_method == 'pointlora':
            nb1, c1 = self.center_divider1(pts)
            nb2, c2 = self.center_divider2(pts)
            t1 = self.encoder(nb1); p1 = self.pos_embed(c1)
            t2 = self.encoder(nb2); p2 = self.pos_embed(c2)
            m1 = self.lora_mask_predictor(t1)
            m2 = self.lora_mask_predictor(t2)
            self._mask_loss = torch.mean(m1 * torch.log(m1 + 1e-6) + (1 - m1) * torch.log(1 - m1 + 1e-6)) \
                            + 0.25 * torch.mean(m2 * torch.log(m2 + 1e-6) + (1 - m2) * torch.log(1 - m2 + 1e-6))
            _, idx1 = torch.topk(m1, self.peft_plora_k1, dim=1)
            bidx1 = torch.arange(t1.size(0), device=t1.device).unsqueeze(1).expand(-1, self.peft_plora_k1)
            tok1, pos1 = t1[bidx1, idx1], p1[bidx1, idx1]
            _, idx2 = torch.topk(m2, self.peft_plora_k2, dim=1)
            bidx2 = torch.arange(t2.size(0), device=t2.device).unsqueeze(1).expand(-1, self.peft_plora_k2)
            tok2, pos2 = t2[bidx2, idx2], p2[bidx2, idx2]
            x = torch.cat([group_input_tokens, tok1, tok2], dim=1)   # [B, G+K1+K2, C]
            pos = torch.cat([pos_base, pos1, pos2], dim=1)
        else:
            x = group_input_tokens

        # === [PointPEFT 新增] point-prior 检索 + 几何索引(每 batch 一次,与分类 port 一致) ===
        if self.peft_method == 'pointpeft':
            if self.bank is None or self.bank.numel() == 0:
                raise RuntimeError('PointPEFT: point-prior bank is not loaded. '
                                   'Run test_pointpeft_seg.py / build_pointpeft_bank.py first.')
            Bsz = pts.size(0)
            ft = self._bank_feature_from_group(neighborhood, center)   # (B, D), 冻结骨干
            prior = self._gen_prompt_prior(ft)                          # (B, K, D)
            prompt_coords_b = self.prompt_coords.unsqueeze(0).expand(Bsz, -1, -1)
            geo_p = self._geo_index(torch.cat([prompt_coords_b, center], dim=1)) \
                if self.peft_prompt_layers > 0 else None
            geo_n = self._geo_index(center) if self.peft_prompt_layers < self.depth else None

        # === Manual block loop with skip connection ===
        if self.prompt_tuning is not None:
            x = self.prompt_tuning(x)          # [B, G+P, C]
        skip_layer = self.depth // 2  # e.g., layer 6 of 12
        feat_mid = None

        for i, block in enumerate(self.blocks.blocks):
            # === [PointPEFT 新增] prompt 前置 + block + GA-Adapter + prompt 剥离 ===
            if self.peft_method == 'pointpeft':
                if i < self.peft_prompt_layers:
                    R = self.prompt_tokens[i].unsqueeze(0).expand(Bsz, -1, -1)   # (B,K,D)
                    P = R + prior
                    x = torch.cat([P, x], dim=1)                                  # (B,K+G,D)
                    pos_i = torch.cat([torch.zeros_like(pos_base[:, :self.peft_prompt_len]), pos], dim=1)
                    x = block(x + pos_i)
                    x = self.ga_adapters[i](x, geo_p, self.group_attn)
                    x = x[:, self.peft_prompt_len:]                               # 剥离 prompt
                else:
                    x = block(x + pos)
                    x = self.ga_adapters[i](x, geo_n, self.group_attn)
                if i == skip_layer - 1:
                    feat_mid = x.clone()
                continue
            if self.hpe_module is not None and self.peft_per_layer and self.layer_hpe_scale is not None:
                layer_mod = hpe_out * self.layer_hpe_scale[i] + self.layer_hpe_bias[i]
                pos_i = pos_base + layer_mod
            elif self.hpe_module is not None:
                pos_i = pos
            else:
                pos_i = pos    # pointlora: 已拼接完整 pos; 其余分支: pos == pos_base

            if self.prompt_tuning is not None:
                pos_i = torch.cat([torch.zeros_like(pos_base[:, :self.peft_num_prompts]), pos_i], dim=1)
            x = block(x + pos_i)
            if i == skip_layer - 1:  # Save mid-level features
                feat_mid = x.clone()
                if self.prompt_tuning is not None:
                    feat_mid = feat_mid[:, self.peft_num_prompts:]  # [B, G, C]

        x = self.norm(x)                     # [B, G(+P)(+K1+K2), C]
        if self.prompt_tuning is not None:
            x = x[:, self.peft_num_prompts:]   # [B, G, C]
        # === [PointLoRA 新增] 剥离尾部 selected tokens ===
        if self.peft_method == 'pointlora':
            x = x[:, :self.num_group]
        feat_mid = self.norm(feat_mid)       # [B, G(+...), C] mid features
        # === [PointLoRA 新增] ===
        if self.peft_method == 'pointlora':
            feat_mid = feat_mid[:, :self.num_group]

        # === Fuse mid + final features ===
        x_cat = torch.cat([feat_mid, x], dim=-1)  # [B, G, 2C]
        x_fused = self.skip_fusion(x_cat)         # [B, G, C]

        # Feature Propagation: upsample from G centers to N points
        x_fp = self.fp(x_fused, center, xyz)  # [B, C, N]

        # === [优化] 解码器位置编码注入 ===
        xyz_t = xyz.permute(0, 2, 1)           # [B, 3, N]
        dec_pos = self.decoder_pos_embed(xyz)  # [B, N, D]
        dec_pos = dec_pos.permute(0, 2, 1)     # [B, D, N]
        x_fp = x_fp + 0.1 * dec_pos            # 小权重注入，不破坏已有特征

        # Segmentation head
        seg_logits = self.seg_head(x_fp)  # [B, 50, N]

        return seg_logits

    def get_loss_acc(self, logits, target):
        """
        [优化] CE + Dice 联合损失，兼顾全局分类精度和边界分割质量。
        """
        B, C, N = logits.shape
        ce_loss = self.loss_ce(logits.reshape(B, C, -1), target.reshape(B, -1))

        if self.dice_weight > 0:
            dice_loss_val = self._dice_loss(logits, target)
            loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss_val
        else:
            loss = ce_loss

        # === [PointLoRA 新增] mask 正则项(官方 lambda=0.004) ===
        if self.peft_method == 'pointlora' and hasattr(self, '_mask_loss'):
            loss = loss + 0.004 * self._mask_loss

        pred = logits.argmax(dim=1)
        acc = (pred == target).sum().float() / target.numel()
        return loss, acc * 100
