import torch
import torch.nn as nn
import torch.nn.functional as F

class HPE_PEFT(nn.Module):
    """
    Hierarchical Positional Encoding as PEFT module.
    
    Takes grouped point neighborhoods and produces geometry-aware 
    positional embeddings. Only this module is trainable; backbone frozen.
    
    Input:  neighborhood [B, G, K, 3] (offsets relative to center)
            center       [B, G, 3]
    Output: pos_embed    [B, G, trans_dim]
    """
    def __init__(self, group_size=32, trans_dim=384, bottleneck=64):
        super().__init__()
        
        # === Per-point geometric feature extractor ===
        # Input: [B*G, 10, K] (channels=10, length=K)
        # Use Conv1d instead of Linear (operates on channel dim, compatible with BN1d)
        self.local_encoder = nn.Sequential(
            nn.Conv1d(10, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
            nn.Conv1d(bottleneck, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
        )
        
        # === Cross-point aggregation (lightweight attention) ===
        self.attn_score = nn.Sequential(
            nn.Conv1d(bottleneck, bottleneck // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(bottleneck // 2, 1, kernel_size=1),
        )
        
        # === Hierarchical refinement ===
        self.group_refine = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
        )
        
        # === Project to transformer dimension ===
        self.proj = nn.Linear(bottleneck, trans_dim)
        
        # Zero-init for stable training (start from zero perturbation)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        # === Multi-scale features ===
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)
    
    def _compute_geometric_features(self, neighborhood, center):
        """
        Extract rich geometric features from each group.
        
        Args:
            neighborhood: [B, G, K, 3] - offsets relative to center
            center: [B, G, 3] - group centers
        Returns:
            features: [B, G, K, 10] per-point geometric features
        """
        B, G, K, _ = neighborhood.shape
        
        # 1. Local coordinates (already centered)
        local_xyz = neighborhood  # [B, G, K, 3]
        
        # 2. Distance to group center
        dist_to_center = torch.norm(local_xyz, dim=-1, keepdim=True)  # [B, G, K, 1]
        
        # 3. Per-group statistics (broadcast to each point)
        local_mean = local_xyz.mean(dim=2, keepdim=True).expand(-1, -1, K, -1)   # [B, G, K, 3]
        local_std = local_xyz.std(dim=2, keepdim=True).expand(-1, -1, K, -1) + 1e-8  # [B, G, K, 3]
        
        # Concatenate: [x, y, z, dist, mean_x, mean_y, mean_z, std_x, std_y, std_z]
        features = torch.cat([local_xyz, dist_to_center, local_mean, local_std], dim=-1)
        
        return features  # [B, G, K, 10]
    
    def forward(self, neighborhood, center):
        """
        Args:
            neighborhood: [B, G, K, 3] - grouped point offsets
            center: [B, G, 3] - group centers
        Returns:
            pos_embed: [B, G, trans_dim] - hierarchical positional embeddings
        """
        B, G, K, _ = neighborhood.shape
        
        # 1. Extract geometric features
        geo_feats = self._compute_geometric_features(neighborhood, center)  # [B, G, K, 10]
        
        # 2. Reshape for Conv1d processing: [B*G, C, K] where C=10
        geo_feats = geo_feats.reshape(B * G, K, 10)  # [B*G, K, 10]
        geo_feats = geo_feats.permute(0, 2, 1)        # [B*G, 10, K]
        
        # 3. Per-point encoding via Conv1d
        point_feats = self.local_encoder(geo_feats)    # [B*G, bottleneck, K]
        
        # 4. Attention-based aggregation across K points
        attn = self.attn_score(point_feats)            # [B*G, 1, K]
        attn = F.softmax(attn, dim=2)                  # [B*G, 1, K]
        group_feats = (point_feats * attn).sum(dim=2)  # [B*G, bottleneck]
        
        # 5. Group-level refinement
        group_feats = self.group_refine(group_feats)   # [B*G, bottleneck]
        
        # 6. Project to transformer dimension
        pos_embed = self.proj(group_feats)             # [B*G, trans_dim]
        pos_embed = pos_embed.reshape(B, G, -1)        # [B, G, trans_dim]
        
        return pos_embed
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
