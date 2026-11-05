import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
import numpy as np

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from .peft import freeze_backbone
from utils.lora_pointlora import LoRALinear
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
# from utils.lora import TrAdapter

from torch.utils.checkpoint import checkpoint


class MaskPredictor(nn.Module):
    def __init__(self, token_dim):
        super(MaskPredictor, self).__init__()
        self.fc1 = nn.Linear(token_dim, token_dim // 2)
        self.fc2 = nn.Linear(token_dim // 2, 1)
        
    def forward(self, tokens):
        """
        input: tokens (B, num_tokens, token_dim)
        output: mask_prob (B, num_tokens)
        """
        x = F.relu(self.fc1(tokens))
        x = self.fc2(x).squeeze(-1)
        mask_prob = torch.sigmoid(x)
        return mask_prob


class Encoder(nn.Module):
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
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
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
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) 
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) 
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
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

    def forward(self, x, prompt_mlp_proj, prompt_mlp_qkv, point_nn=None):
        B, N, C = x.shape
        if hasattr(self.qkv, 'linear'):
            qkv = self.qkv(x, prompt_mlp_qkv)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if point_nn is not None:
            point_nn = point_nn.reshape(B, point_nn.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat([point_nn, k], dim=-2)
            v = torch.cat([point_nn, v], dim=-2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        if hasattr(self.proj, 'linear'):
            x = self.proj(x, prompt_mlp_proj)
        else:
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

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, prompt_mlp_proj, prompt_mlp_qkv):
        x = x + self.drop_path(self.attn(self.norm1(x), prompt_mlp_proj, prompt_mlp_qkv))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.num_gruop = 128
        self.num_gruop_center = 40

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        
        self.prompt_mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
        )
        
        self.prompt_pos_mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
        )

        self.prompt_proj_mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
        )

        self.prompt_qkv_mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim*3),
        )

        self.prompt_mlp.apply(self._init_weights)
        self.prompt_pos_mlp.apply(self._init_weights)
        self.prompt_proj_mlp.apply(self._init_weights)
        self.prompt_qkv_mlp.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = x + pos
            x[:, 1:self.num_gruop+1] = x[:, 1:self.num_gruop+1] + self.prompt_mlp(x[:, 1:self.num_gruop+1])
            x[:, -self.num_gruop_center:] = x[:, -self.num_gruop_center:] + self.prompt_pos_mlp(x[:, -self.num_gruop_center:])
            x = block(x, self.prompt_proj_mlp, self.prompt_qkv_mlp)
        
        return x


@MODELS.register_module()
class PointTransformer_PointLoRA(nn.Module):
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

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.center_divider1 = Group(num_group=128, group_size=32)
        self.center_divider2 = Group(num_group=64, group_size=64)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.lora_mask_predictor = MaskPredictor(token_dim=self.trans_dim)
        self.threshold = 0.4

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

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

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long()) + 0.004 * self._mask_loss
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

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
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)
        self._inject_lora_and_freeze()

    def _inject_lora_and_freeze(self):
        """ckpt 加载完成后注入 LoRA（官方顺序），再冻结骨干"""
        for block in self.blocks.blocks:
            block.attn.qkv = LoRALinear(8, 8, block.attn.qkv)
            block.attn.proj = LoRALinear(8, 8, block.attn.proj)
        freeze_backbone(self, exclude_keywords=[
            'cls_head_finetune', 'cls_token', 'cls_pos',
            'mask_predictor', 'lora', 'prompt_',
            'qkv.weight', 'proj.weight'])

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

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
 
        neighbor_center1, center_center1 = self.center_divider1(pts)
        neighbor_center2, center_center2 = self.center_divider2(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N -> B G C

        center_input_tokens1 = self.encoder(neighbor_center1)
        center_input_tokens2 = self.encoder(neighbor_center2)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        center_pos1 = self.pos_embed(center_center1)
        center_pos2 = self.pos_embed(center_center2)

        mask_prob1 = self.lora_mask_predictor(center_input_tokens1)
        mask_prob2 = self.lora_mask_predictor(center_input_tokens2)

        mask_loss = torch.mean(mask_prob1 * torch.log(mask_prob1 + 1e-6) + (1 - mask_prob1) * torch.log(1 - mask_prob1 + 1e-6))
        mask_loss += 0.25 * torch.mean(mask_prob2 * torch.log(mask_prob2 + 1e-6) + (1 - mask_prob2) * torch.log(1 - mask_prob2 + 1e-6))

        # for scale1
        K = 32  # selected token number
        _, topk_indices = torch.topk(mask_prob1, K, dim=1)
        batch_indices = torch.arange(center_input_tokens1.size(0), device=center_input_tokens1.device).unsqueeze(1).expand(-1, K)  # 生成 batch 索引
        compressed_tokens1 = center_input_tokens1[batch_indices, topk_indices]  
        compressed_pos1 = center_pos1[batch_indices, topk_indices]

        # for scale2
        K = 8  # selected token number
        _, topk_indices = torch.topk(mask_prob2, K, dim=1)
        batch_indices = torch.arange(center_input_tokens2.size(0), device=center_input_tokens2.device).unsqueeze(1).expand(-1, K)  # 生成 batch 索引
        compressed_tokens2 = center_input_tokens2[batch_indices, topk_indices]  
        compressed_pos2 = center_pos2[batch_indices, topk_indices]

        x = torch.cat((cls_tokens, group_input_tokens, compressed_tokens1, compressed_tokens2), dim=1)
        pos = torch.cat((cls_pos, pos, compressed_pos1, compressed_pos2), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        self._mask_loss = mask_loss
        return ret
