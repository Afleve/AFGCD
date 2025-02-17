import torch
import torch.nn as nn

from models.TIME import Token_Importance_Measurer

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, aux_preds = x
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        
        logits = self.last_layer(x)
        return x_proj, logits, aux_preds

class TokenAdaptivePruner(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        
        self.num_labeled_classes = args.num_labeled_classes
        self.feat_dim = args.feat_dim
        self.threshold = args.threshold
        self.pretrainModel = backbone
        if not isinstance(args.image_size, tuple):
            self.mask_length = int(args.image_size / 16)
        else:
            self.mask_length = int(args.image_size[0] / 16)
        self.batch_size = args.batch_size
        
        self.norm = nn.LayerNorm(args.feat_dim, eps=1e-6)

        self.init_TIME()
        self.set_remain_token_num(0)

        self.flip_flag = 0
    
    def get_remain_token_num(self):
        return self.remain_token_num
    
    def set_remain_token_num(self, num):
        self.remain_token_num = num

    def init_TIME(self):
        self.TIME = nn.ModuleList([   
                    Token_Importance_Measurer(num_classes=self.num_labeled_classes, feat_dim=self.feat_dim) for _ in range(len(self.pretrainModel.blocks) - 1)
                ])

    def forward(self, data):
        imgs, training = data
        preds = []
        attn_scores = []
        
        x = self.pretrainModel.prepare_tokens(imgs)
        for i, blk in enumerate(self.pretrainModel.blocks):
            if i < len(self.pretrainModel.blocks) - 1:
                x = self.block_forward(blk, x)
                # All Layer TIME
                attn_score, pred = self.TIME[i](x.detach())
                preds.append(pred)
                attn_scores.append(attn_score)
            else:
                last_x = x
                token_masks = self.prune(attn_scores, training)
                last_x_with_mask = self.block_forward(blk, last_x, token_masks)
                last_x_with_mask = self.pretrainModel.norm(last_x_with_mask)
                # prune token not remain
                mask_num = token_masks.sum(1).unsqueeze(-1)
                last_x = (last_x_with_mask * token_masks.unsqueeze(-1).expand_as(last_x_with_mask)).sum(1) / mask_num
                x = self.norm(last_x)
                
        return x, preds

    def prune(self, scores, training):

        patch_scores = torch.cat([score[:, 1:].unsqueeze(1) for score in scores], dim=1)
        patch_scores = patch_scores.softmax(-1).mean(1)
 
        patch_sort, patch_idx_sort = torch.sort(patch_scores, dim=1, descending=False)
        patch_cum = torch.cumsum(patch_sort, dim=1)
        patch_masks = patch_cum > (self.threshold)

        new_masks = torch.zeros_like(patch_masks).bool().cuda()
        rows, _ = torch.where(patch_masks)
        selected_idxs = patch_idx_sort[patch_masks]
        new_masks[rows, selected_idxs] = True
        patch_masks = new_masks

        # remain one view all Patches
        if training:
            patch_masks[self.batch_size:] = True

        remain_token_num = patch_masks[:self.batch_size].sum()

        self.set_remain_token_num(self.get_remain_token_num() + remain_token_num)

        unit_mask = torch.ones(patch_masks.shape[0], 1).bool().cuda()
        token_masks = torch.cat([unit_mask, patch_masks], dim=1)

        return token_masks

    def block_forward(self, blk, x, token_mask=None, return_attn=False):
        if token_mask is not None:
            y, attn = self.attn_forward(blk.attn, blk.norm1(x), token_mask)
        else:
            y, attn = blk.attn(blk.norm1(x))

        x = x + blk.drop_path(y)
        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
        
        if return_attn:
            return x, attn
        return x

    def attn_forward(self, attnModel, x, token_mask):
        B, N, C = x.shape
        qkv = attnModel.qkv(x).reshape(B, N, 3, attnModel.num_heads, C // attnModel.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * attnModel.scale

        mask = token_mask.unsqueeze(2).float() @ token_mask.unsqueeze(1).float()
        mask = mask.unsqueeze(1).repeat(1, attn.shape[1], 1, 1).bool()
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = attn.masked_fill(~mask, 0.0)

        attn = attnModel.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attnModel.proj(x)
        x = attnModel.proj_drop(x)
        return x, attn
    

    