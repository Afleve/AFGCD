import torch
import torch.nn as nn
from timm.layers.mlp import Mlp
from timm.models.layers import trunc_normal_
from timm.layers.weight_init import trunc_normal_tf_

class Simple_Cross_Attention(nn.Module):
    def __init__(self, feat_dim=768, mlp_ratio=4.0):
        super().__init__()
        
        self.D_sqrt = feat_dim**-0.5
        self.learnable_query = nn.Parameter(torch.empty(1, feat_dim))
        self.mlp_norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.mlp = Mlp(feat_dim, int(feat_dim * mlp_ratio))

        trunc_normal_tf_(self.learnable_query, std=self.D_sqrt)

    def forward(self, x):
        B = x.shape[0]
        # QKV
        q = self.learnable_query.repeat(B, 1).unsqueeze(1)
        k, v = x, x
        attn = q @ k.transpose(-2, -1)
        attn_sm = (attn * self.D_sqrt).softmax(dim=-1)
        x = (attn_sm @ v).squeeze(1)
        # FFN
        x = x + self.mlp(self.mlp_norm(x))
        return x, attn.squeeze(1)

class Aux_Head(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)
        trunc_normal_(self.head.weight, std=0.02)
        self.head.bias.data.zero_()

    def forward(self, x):
        x = self.norm(x)
        x = self.head(x)
        return x

class Token_Importance_Measurer(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=768):
        super().__init__()
        self.sim_cross_attn = Simple_Cross_Attention(feat_dim=feat_dim)
        self.aux_head = Aux_Head(feat_dim, num_classes)

    def forward(self, x):
        weighted_feat, attn = self.sim_cross_attn(x)
        pred = self.aux_head(weighted_feat)
        return attn, pred
