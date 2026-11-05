import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from knn_cuda import KNN
from .peft import freeze_backbone


class Encoder(nn.Module):
    """mini-PointNet token embed: B G k 3 -> B G C"""
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)  # B G 3
        _, idx = self.knn(xyz, center)          # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerBlocks(nn.Module):
    """标准 block 序列；属性名 blocks 与预训练 ckpt 键路径 blocks.blocks.* 对齐。"""
    def __init__(self, dim, depth, dpr, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  drop=drop, attn_drop=attn_drop,
                  drop_path=dpr[i] if isinstance(dpr, list) else dpr)
            for i in range(depth)])


class GroupAttention(nn.Module):
    """GA-Adapter 内的低维共享组内注意力（跨所有 block 共享一份权重）。"""
    def __init__(self, dim, num_heads=6, low_dim=18, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.low_dim = low_dim
        head_dim = low_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, low_dim * 3, bias=False)
        self.proj = nn.Linear(low_dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.low_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.low_dim)
        x = self.proj(x)
        return x


class GAAdapter(nn.Module):
    """Geometry-aware Adapter：bottleneck MLP -> FPS 中心 -> kNN 邻域
    -> 共享组内注意力 -> pooling+alpha 融合 -> 反距离传播+beta 融合 -> 输出 MLP"""
    def __init__(self, dim, bottleneck=16, drop=0.):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 中心特征融合系数（论文 alpha）
        self.beta = nn.Parameter(torch.tensor(0.2))   # 传播融合系数（论文 beta）
        self.drop = nn.Dropout(drop)

    def forward(self, x, geo, group_attn):
        # x: (B, Nt, D)，geo 为预计算的 (cen_idx, nn_idx, flat_idx, flat_w, inv)
        B, Nt, D = x.shape
        T = self.up(F.gelu(self.down(x)))                 # (B,Nt,D)
        cen_idx, nn_idx, flat_idx, flat_w, inv = geo
        ar = torch.arange(B, device=x.device)
        T_c = T[ar[:, None], cen_idx]                     # (B,Nc,D)
        T_n = T[ar[:, None, None], nn_idx]                # (B,Nc,k,D)
        Nc, k = nn_idx.shape[1], nn_idx.shape[2]
        T_n = group_attn(T_n.reshape(B * Nc, k, D)).reshape(B, Nc, k, D)
        pooled = T_n.max(dim=2)[0] + T_n.mean(dim=2)      # (B,Nc,D)
        center_enh = pooled + self.alpha * T_c            # (B,Nc,D)
        feat = center_enh.unsqueeze(2).expand(-1, -1, k, -1).reshape(B, Nc * k, D)
        f_sum = torch.zeros(B, Nt, D, device=x.device).scatter_add_(
            1, flat_idx.unsqueeze(-1).expand(-1, -1, D), flat_w.unsqueeze(-1) * feat)
        interp = f_sum * inv.unsqueeze(-1)                # (B,Nt,D)
        T = T + self.beta * interp
        out = self.out_mlp(T)
        return x + self.drop(out)


@MODELS.register_module()
class PointTransformer_PointPEFT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.prompt_layers = int(getattr(config, 'prompt_layers', 12))
        self.prompt_len = int(getattr(config, 'prompt_len', 16))
        self.adapter_bottleneck = int(getattr(config, 'adapter_bottleneck', 16))
        self.ga_n_center = int(getattr(config, 'ga_n_center', 32))
        self.ga_knn = int(getattr(config, 'ga_knn', 16))
        self.bank_path = getattr(config, 'bank_path', None)
        assert 0 < self.prompt_layers <= self.depth
        assert self.prompt_len > 2

        # ---------- 主干（键路径与预训练 ckpt 对齐） ----------
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerBlocks(dim=self.trans_dim, depth=self.depth, dpr=dpr,
                                        num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.trans_dim)

        # ---------- 冻结特征提取器（point-prior bank 的编码器，独立副本） ----------
        self.bank_encoder = Encoder(encoder_channel=self.encoder_dims)
        self.bank_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.bank_blocks = TransformerBlocks(dim=self.trans_dim, depth=self.depth, dpr=dpr,
                                             num_heads=self.num_heads)
        self.bank_norm = nn.LayerNorm(self.trans_dim)

        # ---------- P2-Prompt ----------
        # prompt_tokens[i] 为第 i 层可学习 token（论文 R_i），形状 (L, K, D)
        self.prompt_tokens = nn.Parameter(torch.zeros(self.prompt_layers, self.prompt_len, self.trans_dim))
        # prompt token 的可学习 3D 坐标（参与 GA-Adapter 的 FPS/kNN）
        self.prompt_coords = nn.Parameter(torch.zeros(self.prompt_len, 3))
        # P_prior 的 bottleneck MLP（逐 token 变换 Concat(FT, FA, X_{K-2})）
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.trans_dim, self.adapter_bottleneck),
            nn.GELU(),
            nn.Linear(self.adapter_bottleneck, self.trans_dim),
        )

        # ---------- GA-Adapter（每层一份，组内注意力跨层共享） ----------
        self.group_attn = GroupAttention(dim=self.trans_dim, num_heads=self.num_heads)
        self.ga_adapters = nn.ModuleList([
            GAAdapter(dim=self.trans_dim, bottleneck=self.adapter_bottleneck)
            for _ in range(self.depth)
        ])

        # ---------- 分类头（mean+max pooling 拼接，与全表方法同口径） ----------
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.register_buffer('bank', torch.zeros(0), persistent=False)
        self.knn_geo = KNN(k=self.ga_knn, transpose_mode=True)

        self.build_loss_func()
        trunc_normal_(self.prompt_tokens, std=.02)
        trunc_normal_(self.prompt_coords, std=.02)
        self.prior_mlp.apply(self._init_weights)

        if self.bank is None or self.bank.numel() == 0:
            self._load_bank()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ---------- 记忆库 ----------
    def _load_bank(self):
        path = self.bank_path
        if path and os.path.exists(path):
            bank = torch.load(path, map_location='cpu')
            bank = F.normalize(bank.float(), dim=-1)
            self.register_buffer('bank', bank, persistent=False)
            print_log(f'[PointPEFT] Loaded point-prior bank from {path}, shape {tuple(bank.shape)}',
                      logger='PointPEFT')
        else:
            print_log(f'[PointPEFT] WARNING: point-prior bank not found at {path}. '
                      f'Run build_pointpeft_bank.py first; forward() will raise until bank exists.',
                      logger='PointPEFT')

    def _set_bank_eval(self):
        self.bank_encoder.eval()
        self.bank_blocks.eval()
        self.bank_norm.eval()

    def _bank_feature_from_group(self, neighborhood, center):
        """冻结预训练骨干前向：patch tokens mean pooling -> (B, D)"""
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
        """外部（bank 生成脚本）使用的特征提取接口"""
        neighborhood, center = self.group_divider(pts)
        return self._bank_feature_from_group(neighborhood, center)

    def _gen_prompt_prior(self, ft):
        """论文式(6)-(8)：cosine 检索 top-(K-2) -> softmax 聚合 -> concat -> bottleneck MLP"""
        K = self.prompt_len
        q = F.normalize(ft, dim=-1)                    # (B,D)
        S = q @ self.bank.t()                          # (B,Ntr)
        topv, topi = torch.topk(S, K - 2, dim=-1)      # (B,K-2)
        Xk = self.bank[topi]                           # (B,K-2,D)
        w = F.softmax(topv, dim=-1).unsqueeze(-1)      # (B,K-2,1)
        FA = (w * Xk).sum(dim=1, keepdim=True)         # (B,1,D)
        inp = torch.cat([q.unsqueeze(1), FA, Xk], dim=1)  # (B,K,D)
        return self.prior_mlp(inp)                     # (B,K,D)

    # ---------- 几何索引（与层无关，每 batch 每类只算一次） ----------
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
        """返回 (cen_idx, nn_idx, flat_idx, flat_w, inv)，全部无梯度"""
        with torch.no_grad():
            B, Nt, _ = coords.shape
            cen_idx = self._fps_index(coords, self.ga_n_center)   # (B,Nc)
            ar = torch.arange(B, device=coords.device)
            coords_c = coords[ar[:, None], cen_idx]               # (B,Nc,3)
            _, nn_idx = self.knn_geo(coords, coords_c)            # (B,Nc,k)
            coords_n = coords[ar[:, None, None], nn_idx]          # (B,Nc,k,3)
            dist2 = ((coords_n - coords_c[:, :, None, :]) ** 2).sum(-1)
            w = 1.0 / (dist2 + 1e-8)
            flat_idx = nn_idx.reshape(B, -1)
            flat_w = w.reshape(B, -1)
            w_sum = torch.zeros(B, Nt, device=coords.device).scatter_add_(1, flat_idx, flat_w)
            inv = 1.0 / (w_sum + 1e-8)
            return cen_idx, nn_idx, flat_idx, flat_w, inv

    # ---------- 权重加载与冻结 ----------
    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(get_missing_parameters_message(incompatible.missing_keys), logger='Transformer')
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='Transformer')
            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

            # bank 提取器独立加载同一份预训练权重
            bank_sd = {}
            for k, v in base_ckpt.items():
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

            # 冻结：仅训练 head / prompt / prior / GA-Adapter / 共享注意力 / 骨干 bias
            freeze_backbone(self, exclude_keywords=[
                'cls_head_finetune', 'prompt', 'prior_mlp', 'ga_adapters',
                'group_attn', 'bias'])
            # bank 提取器整体强制冻结（防止 bias 关键字解冻）
            for n, p in self.named_parameters():
                if n.startswith('bank_'):
                    p.requires_grad_(False)
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def forward(self, pts):
        B = pts.shape[0]
        neighborhood, center = self.group_divider(pts)     # (B,G,32,3), (B,G,3)
        x = self.encoder(neighborhood)                     # (B,G,D)
        x = x + self.pos_embed(center)

        if self.bank is None or self.bank.numel() == 0:
            raise RuntimeError('PointPEFT: point-prior bank is not loaded. '
                               'Run build_pointpeft_bank.py first.')

        # 冻结骨干提取当前样本特征 -> point-prior 检索
        ft = self._bank_feature_from_group(neighborhood, center)
        prior = self._gen_prompt_prior(ft)                 # (B,K,D)

        # 几何索引（prompt 层 / 非 prompt 层两类）
        prompt_coords = self.prompt_coords.unsqueeze(0).expand(B, -1, -1)
        geo_p = self._geo_index(torch.cat([prompt_coords, center], dim=1)) if self.prompt_layers > 0 else None
        geo_n = self._geo_index(center) if self.prompt_layers < self.depth else None

        for i, blk in enumerate(self.blocks.blocks):
            if i < self.prompt_layers:
                R = self.prompt_tokens[i].unsqueeze(0).expand(B, -1, -1)  # (B,K,D)
                P = R + prior
                x = torch.cat([P, x], dim=1)               # (B,K+G,D)
                x = blk(x)
                x = self.ga_adapters[i](x, geo_p, self.group_attn)
                x = x[:, self.prompt_len:]                 # 剥离 prompt，只留 point tokens
            else:
                x = blk(x)
                x = self.ga_adapters[i](x, geo_n, self.group_attn)

        x = self.norm(x)
        concat_f = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)  # (B,2D)
        ret = self.cls_head_finetune(concat_f)
        return ret
