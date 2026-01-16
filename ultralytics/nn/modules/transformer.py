# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Transformer modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .position_encoding import Conv2dNormActivation,get_sine_pos_embed
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'PositionRelationEmbedding')


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FourierMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=256, sigma=10.0):
        """
        Args:
            input_dim: 输入特征维度 (我们是 7: 1 GCD + 2 Angle + 4 LogRel)
            sigma: 控制频率分布的标准差。
                   sigma 越大，对高频细节(密集小目标)越敏感；
                   sigma 越小，对低频(整体布局)越敏感。
                   对于 UAV 密集场景，建议设为 10.0 - 30.0。
        """
        super().__init__()
        # 1. 随机高斯矩阵 B (不可学习，类似位置编码的基底)
        # 映射到 hidden_dim 的一半，因为后面会有 sin 和 cos 拼起来
        self.mapping_size = hidden_dim // 2
        self.register_buffer('B', torch.randn(input_dim, self.mapping_size) * sigma)
        
        # 2. 后续的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # 输入是 sin + cos
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [BS, N, M, input_dim]
        
        # 1. 傅里叶映射 (Fourier Mapping)
        # v -> [sin(2*pi*B*v), cos(2*pi*B*v)]
        # 这一步把低维坐标映射到了高维流形，且消除了周期性混淆
        projected = (2 * torch.pi * x) @ self.B
        x_fourier = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        
        # 2. MLP 处理
        return self.mlp(x_fourier)

class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()
        self.num_heads = n_heads

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class Deformable_position_TransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = decoder_layer.num_heads
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.position_relation_embedding = PositionRelationEmbedding(4, self.num_heads) 

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, pos_relation, pos_mlp(refer_bbox))
            
            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            if i>0:
                dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:

                    dec_bboxes.append(dec_bbox)
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break
            
            if i == self.num_layers - 1:
                break
            src_boxes = tgt_boxes if i >=1 else refer_bbox
            tgt_boxes = refined_bbox if i==0 else dec_bbox
            pos_relation = self.position_relation_embedding(src_boxes,tgt_boxes).flatten(0,1)

            if attn_mask is not None:
                pos_relation.masked_fill_(attn_mask,float("-inf"))
            
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

    

# ================= Variation B: Original (Relation-DETR) =================
def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    """
    输出维度: 4 (delta_x, delta_y, delta_w, delta_h)
    """
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    
    pos_embed = torch.cat([delta_xy, delta_wh], -1) 
    return pos_embed

# ================= Variation C: Original + IoU =================
def iou_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    """
    输出维度: 5 (Original 4 dims + 1 IoU)
    """
    # 1. 计算基础特征 (Variation B)
    base_embed = box_rel_encoding(src_boxes, tgt_boxes, eps)
    
    # 2. 计算 IoU
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    
    x1y1_1, x2y2_1 = xy1 - wh1/2, xy1 + wh1/2
    x1y1_2, x2y2_2 = xy2 - wh2/2, xy2 + wh2/2
    
    # 广播机制: [BS, N, 1, 2] vs [BS, 1, M, 2]
    x1y1_inter = torch.maximum(x1y1_1.unsqueeze(-2), x1y1_2.unsqueeze(-3))
    x2y2_inter = torch.minimum(x2y2_1.unsqueeze(-2), x2y2_2.unsqueeze(-3))
    
    wh_inter = torch.clamp(x2y2_inter - x1y1_inter, min=0)
    area_inter = wh_inter.prod(dim=-1)
    
    area_1 = wh1.prod(dim=-1).unsqueeze(-1)
    area_2 = wh2.prod(dim=-1).unsqueeze(-2)
    
    iou = area_inter / (area_1 + area_2 - area_inter + eps)
    
    # 3. 拼接
    pos_embed = torch.cat([base_embed, iou.unsqueeze(-1)], -1)
    return pos_embed

# def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
#     # construct position relation
#     xy1, wh1 = src_boxes.split([2, 2], -1)
#     xy2, wh2 = tgt_boxes.split([2, 2], -1)
#     delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#     delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
#     delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
#     # 计算IoU
#     x1y1_1, x2y2_1 = xy1 - wh1/2, xy1 + wh1/2
#     x1y1_2, x2y2_2 = xy2 - wh2/2, xy2 + wh2/2
    
#     x1y1_inter = torch.maximum(x1y1_1.unsqueeze(-2), x1y1_2.unsqueeze(-3))
#     x2y2_inter = torch.minimum(x2y2_1.unsqueeze(-2), x2y2_2.unsqueeze(-3))
    
#     wh_inter = torch.clamp(x2y2_inter - x1y1_inter, min=0)
#     area_inter = wh_inter.prod(dim=-1)
    
#     area_1 = wh1.prod(dim=-1).unsqueeze(-1)
#     area_2 = wh2.prod(dim=-1).unsqueeze(-2)
    
#     iou = area_inter / (area_1 + area_2 - area_inter + eps)
    
#     # 计算相对距离
#     center1 = xy1
#     center2 = xy2
#     # rel_distance = torch.norm(center1.unsqueeze(-2) - center2.unsqueeze(-3), dim=-1)
#     # rel_distance = torch.log(rel_distance / torch.sqrt(wh1.prod(dim=-1)).unsqueeze(-1) + eps)
    
#     # 计算相对角度
#     delta_center = center2.unsqueeze(-3) - center1.unsqueeze(-2)
#     angle = torch.atan2(delta_center[..., 1], delta_center[..., 0])
    
#     # 组合所有特征
#     pos_embed = torch.cat([
#         delta_xy,  # 相对位置 (2)
#         delta_wh,  # 相对尺寸 (2)
#         iou.unsqueeze(-1),  # IoU (1)
#         # rel_distance.unsqueeze(-1),  # 相对距离 (1)
#         angle.unsqueeze(-1),  # 相对角度 (1)
#     ], -1)
#     # pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

#     return pos_embed

def gaussian_relation_encoding(src_boxes, tgt_boxes, eps=1e-5):
    """
    基于高斯 Wasserstein 距离的不确定性关系编码 (Uncertainty-Aware Gaussian Relation)
    创新点: 将 BBox 视为 2D 高斯分布，使用 Wasserstein 距离衡量其相似度，
           解决小目标 IoU 敏感和远距离梯度消失问题。
    
    Args:
        src_boxes (Tensor): [BS, N, 4] (cx, cy, w, h), 值域通常在 0-1 之间
        tgt_boxes (Tensor): [BS, M, 4] (cx, cy, w, h)
    """
    # 1. 维度扩展以构建两两配对矩阵 (Pairwise Matrix)
    # src: [BS, N, 1, 4], tgt: [BS, 1, M, 4]
    if src_boxes.dim() == 3:
        b1 = src_boxes.unsqueeze(2)  
        b2 = tgt_boxes.unsqueeze(1)
    else:
        # 处理可能的特殊输入情况
        b1 = src_boxes.unsqueeze(1)
        b2 = tgt_boxes.unsqueeze(0)

    # 2. 提取高斯参数
    # 均值 mu: 中心点 (cx, cy)
    mu1 = b1[..., :2]
    mu2 = b2[..., :2]
    
    # 标准差 sigma: 宽高的一半 (w/2, h/2)，代表不确定性范围
    # 注意: 这里假设输入框是归一化坐标(0-1)。
    sigma1 = b1[..., 2:] / 2.0
    sigma2 = b2[..., 2:] / 2.0

    # 3. 计算 Wasserstein 距离的平方 (W2^2)
    # 公式: ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2 (Frobenius norm for diagonal covariance)
    
    # 位置差异 (Location Discrepancy)
    xy_distance_sq = torch.sum((mu1 - mu2)**2, dim=-1)
    
    # 尺度/形状差异 (Scale/Shape Discrepancy)
    wh_distance_sq = torch.sum((sigma1 - sigma2)**2, dim=-1)
    
    # 总 Wasserstein 距离
    w2_sq = xy_distance_sq + wh_distance_sq

    # 4. 非线性映射: 将距离转换为“相似度” (Gaussian Similarity)
    # 使用指数核函数。tau 是温度系数，控制对距离的敏感度。
    # 对于归一化坐标(0-1)，建议 tau 取较小值 (如 0.05 - 0.1)。
    # 这里我们引入一个自适应的归一化项 (sigma1*sigma2)，使得度量对尺度相对不敏感
    scale_term = (sigma1.prod(dim=-1).sqrt() + sigma2.prod(dim=-1).sqrt()) + eps
    tau = 0.1 
    # 归一化 Wasserstein Distance (NWD) 的变体
    gaussian_similarity = torch.exp(-w2_sq / (tau * scale_term + eps))

    # 5. 辅助几何特征 (保留方向性)
    # 因为高斯分布是对称的，会丢失“A在B左边”这种方向信息，所以需要保留角度特征
    delta_xy = mu2 - mu1
    angle = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])
    
    # 6. 传统的 Log-space 相对特征 (作为补充，保持特征维度为6)
    # 相对位置
    rel_xy = torch.abs(mu1 - mu2) / (sigma1 + eps) 
    rel_xy = torch.log(rel_xy + 1.0)
    # 相对尺寸
    rel_wh = torch.log((sigma1 + eps) / (sigma2 + eps))

    # === 特征融合 ===
    # 输出维度: 1 (Sim) + 1 (Angle) + 2 (XY) + 2 (WH) = 6
    pos_embed = torch.cat([
        gaussian_similarity.unsqueeze(-1), 
        angle.unsqueeze(-1),
        rel_xy,
        rel_wh
    ], dim=-1) 
    
    return pos_embed
# def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
#     # construct position relation
#     xy1, wh1 = src_boxes.split([2, 2], -1)
#     xy2, wh2 = tgt_boxes.split([2, 2], -1)
#     delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#     delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
#     delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
#     pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

#     return pos_embed
class PosEmbedding(nn.Module):
    # def __init__(self):
    def __init__(self, input_dim=6):
        super().__init__()
        self.linear1 = nn.Linear(6, 12)
        # self.linear1 = nn.Linear(input_dim, 12) # 动态输入维度
        self.linear2 = nn.Linear(12, 24)
        self.relu = nn.GELU()
    #     self._init_weights()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.0,
        scale=100.0,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 6,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        # self.pos_func = functools.partial(
        #     get_sine_pos_embed,
        #     num_pos_feats=embed_dim,
        #     temperature=temperature,
        #     scale=scale,
        #     exchange_xy=False,
        # )
        self.pos_func =PosEmbedding()

    # def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
    def forward(self, src_boxes: torch.Tensor, tgt_boxes: torch.Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # # src_boxes: [batch_size, num_boxes1, 4]
        # # tgt_boxes: [batch_size, num_boxes2, 4]
        # torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        # torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        # with torch.no_grad():
        #     pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
        #     pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        # pos_embed = self.pos_proj(pos_embed)
        with torch.no_grad():
            # ================= [INNOVATION START] =================
            # 使用高斯关系编码 (Gaussian Relation) 替代原始的 Box Relation
            # 原始: pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            # pos_embed = gaussian_relation_encoding(src_boxes, tgt_boxes)
            pos_embed = gcd_relation_encoding(src_boxes, tgt_boxes)
            # ================= [INNOVATION END] ===================
            
            # 接下来的 MLP 映射保持不变
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
            
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()

# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         variation='gcd', # 新增参数: 'none', 'original', 'iou', 'gcd'
#         activation_layer=nn.ReLU,
#         inplace=True,
#     ):
#         super().__init__()
#         self.variation = variation.lower()
        
#         # ================= [打印提示信息 START] =================
#         print("-" * 50)
#         print(f" [GCPRE] Initializing Position Relation Module...")
#         print(f" [GCPRE] Current Mode: {self.variation.upper()}")
        
#         # 1. 根据变体确定输入特征维度 (input_dim) 并打印具体配置
#         if self.variation == 'original':   
#             self.input_dim = 4
#             print(" [GCPRE] Configuration: Variation B (Original Relation-DETR)")
#             print(" [GCPRE] Features: [delta_xy(2), delta_wh(2)]")
            
#         elif self.variation == 'iou':      
#             self.input_dim = 5
#             print(" [GCPRE] Configuration: Variation C (Original + IoU)")
#             print(" [GCPRE] Features: [delta_xy(2), delta_wh(2), iou(1)]")
            
#         elif self.variation == 'gcd':      
#             self.input_dim = 6             
#             print(" [GCPRE] Configuration: Variation D (Ours: GCDecoder)")
#             print(" [GCPRE] Features: [GCD_sim(1), Angle(1), Log_dist(4)]")
            
#         elif self.variation == 'none':     
#             self.input_dim = 0
#             print(" [GCPRE] Configuration: Variation A (RT-DETR Baseline)")
#             print(" [GCPRE] Status: Module DISABLED. Returning None.")
            
#         else:
#             raise ValueError(f"Unknown variation: {self.variation}")
            
#         print(f" [GCPRE] Input Dimension: {self.input_dim}")
#         print("-" * 50)
#         # ================= [打印提示信息 END] ===================

#         # 1. 根据变体确定输入特征维度 (input_dim)
#         if self.variation == 'original':   # Variation B
#             self.input_dim = 4
#         elif self.variation == 'iou':      # Variation C
#             self.input_dim = 5
#         elif self.variation == 'gcd':      # Variation D (Ours)
#             self.input_dim = 6             # 对应 gcd_relation_encoding 的输出维度
#         elif self.variation == 'none':     # Variation A
#             self.input_dim = 0
#         else:
#             raise ValueError(f"Unknown variation: {self.variation}")

#         # Variation A 不需要初始化后续网络
#         if self.variation != 'none':
#             # 2. 初始化 MLP，传入对应的 input_dim
#             self.pos_func = PosEmbedding(input_dim=self.input_dim)

#             # 3. 初始化投影层
#             # PosEmbedding 的输出固定是 24 (Linear2的输出)，但后续代码似乎依赖 embed_dim
#             # 原代码逻辑: self.pos_proj 接收 embed_dim * 6 ?? 
#             # 修正: 原代码 self.pos_proj 接收的是 pos_func 的输出。
#             # 你的 PosEmbedding 输出是 24，所以这里 proj 的输入应该是 24。
#             # 但如果你的原意是 pos_func 输出 embed_dim 相关的维度，请调整 PosEmbedding。
#             # 假设按照你提供的 PosEmbedding，输出是 24:
#             self.pos_proj = Conv2dNormActivation(
#                 24,  # 这里必须匹配 PosEmbedding 的输出维度 (self.linear2 的 out_features)
#                 num_heads,
#                 kernel_size=1,
#                 inplace=inplace,
#                 norm_layer=None,
#                 activation_layer=activation_layer,
#             )

#     def forward(self, src_boxes: torch.Tensor, tgt_boxes: torch.Tensor = None):
#         # Variation A: RT-DETR Decoder (不使用位置关系编码)
#         if self.variation == 'none':
#             return None # 或者返回 torch.zeros(...)，取决于你在 Decoder 里的调用方式

#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
            
#         with torch.no_grad():
#             # 根据配置选择特征计算函数
#             if self.variation == 'original':
#                 pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
#             elif self.variation == 'iou':
#                 pos_embed = iou_rel_encoding(src_boxes, tgt_boxes)
#             elif self.variation == 'gcd':
#                 # 使用你提供的 gcd 函数
#                 # 注意：确保 gcd_relation_encoding 在此作用域可见
#                 pos_embed = gcd_relation_encoding(src_boxes, tgt_boxes) 
            
#         # MLP 映射 [BS, N, M, input_dim] -> [BS, N, M, 24]
#         pos_embed = self.pos_func(pos_embed)
        
#         # 调整维度 [BS, 24, N, M] 以适配 Conv2d
#         pos_embed = pos_embed.permute(0, 3, 1, 2)
        
#         # 投影到 Head 维度 [BS, num_heads, N, M]
#         pos_embed = self.pos_proj(pos_embed)

#         return pos_embed.clone()


###
# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         temperature=10000.0,
#         scale=100.0,
#         activation_layer=nn.ReLU,
#         inplace=True,
#     ):
#         super().__init__()
        
#         # 注意：这里的 input_channels 取决于 MLP 的输出
#         # 你的代码里写的是 embed_dim * 6，我们保持一致，让 MLP 输出这个维度
#         proj_input_dim = embed_dim * 6
        
#         self.pos_proj = Conv2dNormActivation(
#             proj_input_dim, 
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
        
#         # ================= [INNOVATION 1: 结构升级] =================
#         # 使用 FourierMLP 替换原来的 PosEmbedding
#         # input_dim=7 (对应 GCD 编码的 7 个通道)
#         # output_dim=proj_input_dim (确保和 pos_proj 对齐)
#         self.pos_func = FourierMLP(
#             input_dim=7, 
#             hidden_dim=embed_dim, 
#             output_dim=proj_input_dim,
#             sigma=20.0  # 针对 UAV 密集场景推荐值
#         )
#         # ================= [INNOVATION END] =======================

#     def forward(self, src_boxes: torch.Tensor, tgt_boxes: torch.Tensor = None):
#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
            
#         with torch.no_grad():
#             # ================= [INNOVATION 2: 几何先验] =================
#             # 使用我们刚才定下来的“稳健版 GCD 编码”
#             # 输出维度应该是 7 (1 Sim + 2 Angle + 4 LogRel)
#             pos_embed = gcd_relation_encoding(src_boxes, tgt_boxes)
#             # ================= [INNOVATION END] =========================
            
#             # 使用 FourierMLP 映射到高维
#             # [BS, N, M, 7] -> [BS, N, M, embed_dim*6]
#             pos_embed = self.pos_func(pos_embed)
            
#             # 调整维度以适配卷积层: [BS, embed_dim*6, N, M]
#             pos_embed = pos_embed.permute(0, 3, 1, 2)
            
#         # 卷积投影，生成最终的 Attention Bias
#         pos_embed = self.pos_proj(pos_embed)

#         return pos_embed.clone()





def gcd_relation_encoding(src_boxes, tgt_boxes, eps=1e-7):
    """
    基于论文原文代码改编的 GCD 关系编码 (Strict Implementation)
    
    逻辑来源: 用户提供的 gcd_loss 函数
    核心思想: 双向相对距离平均 (Symmetrized Relative Distance)
    
    Args:
        src_boxes (Tensor): [BS, N, 4] (cx, cy, w, h)
        tgt_boxes (Tensor): [BS, M, 4] (cx, cy, w, h)
    Returns:
        pos_embed (Tensor): [BS, N, M, 6]
    """
    # 1. 维度对齐与广播 (Broadcasting)
    # src: [BS, N, 1, 4], tgt: [BS, 1, M, 4]
    if src_boxes.dim() == 3:
        b1 = src_boxes.unsqueeze(2)  
        b2 = tgt_boxes.unsqueeze(1)
    else:
        b1 = src_boxes.unsqueeze(1)
        b2 = tgt_boxes.unsqueeze(0)

    # 2. 提取参数 (假设输入已经是 cx, cy, w, h 格式)
    # center1, center2
    cx1, cy1, w1, h1 = b1.unbind(-1)
    cx2, cy2, w2, h2 = b2.unbind(-1)

    # 3. 计算基础差值
    # whs (center distance): dx, dy
    dx = cx1 - cx2
    dy = cy1 - cy2
    
    # 4. 按照原文逻辑计算四项距离
    
    # --- Part 1: Relative to Box 1 (src) ---
    # center_distance1 = (dx/w1)^2 + (dy/h1)^2
    c_dist1 = (dx / (w1 + eps))**2 + (dy / (h1 + eps))**2
    
    # wh_distance2 (注意: 原代码中 wh_distance2 分母是 w1/h1)
    # wh_dist_relative_to_1 = ((w1-w2)/w1)^2 + ((h1-h2)/h1)^2
    # 原文代码 wh_distance2 还有个 /4
    wh_dist1 = (((w1 - w2) / (w1 + eps))**2 + ((h1 - h2) / (h1 + eps))**2) / 4.0

    # --- Part 2: Relative to Box 2 (tgt) ---
    # center_distance2 = (dx/w2)^2 + (dy/h2)^2
    c_dist2 = (dx / (w2 + eps))**2 + (dy / (h2 + eps))**2
    
    # wh_distance1 (注意: 原代码中 wh_distance1 分母是 w2/h2)
    # wh_dist_relative_to_2 = ((w1-w2)/w2)^2 + ((h1-h2)/h2)^2
    wh_dist2 = (((w1 - w2) / (w2 + eps))**2 + ((h1 - h2) / (h2 + eps))**2) / 4.0

    # 5. 组合 GCD (Squared)
    # gcd_2 = (center_distance1 + wh_distance1 + center_distance2 + wh_distance2) / 2
    # 注意对应关系：原代码的变量名下标和分母下标是交叉的，这里我们直接加总平均即可
    gcd_2 = (c_dist1 + wh_dist1 + c_dist2 + wh_dist2) / 2.0

    # 6. 转换为相似度 (Similarity Mode)
    # 对应原代码: if mode == 'exp': gcd = torch.exp(-torch.sqrt(gcd_2))
    # 这是一个 0~1 的值，非常适合作为 Attention 的权重或特征
    gcd_similarity = torch.exp(-torch.sqrt(gcd_2 + eps))

    # 7. 保留辅助几何特征 (角度 + 相对位置)
    # 因为 GCD 是对称标量，丢失了方向信息，所以必须保留 Angle
    # angle = torch.atan2(dy, dx)
    angle = torch.atan2(dy, dx)   # 归一化到 -1 到 1 之间
    
    # 传统的 Log-space 相对特征 (作为补充)
    # 依然保留，给 MLP 提供原始的相对位置信息
    rel_x = torch.log(torch.abs(dx) / (w1 + eps) + 1.0)
    rel_y = torch.log(torch.abs(dy) / (h1 + eps) + 1.0)
    # rel_x = torch.sign(dx) * torch.log(torch.abs(dx) / (w1 + eps) + 1.0)
    # rel_y = torch.sign(dy) * torch.log(torch.abs(dy) / (h1 + eps) + 1.0) # 这里的逻辑是：保持数值也是 log 缩放的，但是保留正负号
    rel_w = torch.log((w1 + eps) / (w2 + eps))
    rel_h = torch.log((h1 + eps) / (h2 + eps))

    # === 特征融合 (6维) ===
    # 1(GCD Sim) + 1(Angle) + 4(Rel Log Coords)
    pos_embed = torch.cat([
        gcd_similarity.unsqueeze(-1), 
        angle.unsqueeze(-1),
        rel_x.unsqueeze(-1), 
        rel_y.unsqueeze(-1),
        rel_w.unsqueeze(-1),
        rel_h.unsqueeze(-1)
    ], dim=-1) 
    
    return pos_embed


# def gcd_relation_encoding(src_boxes, tgt_boxes, eps=1e-7):
#     # 1. 维度对齐 (保持不变)
#     if src_boxes.dim() == 3:
#         b1 = src_boxes.unsqueeze(2)
#         b2 = tgt_boxes.unsqueeze(1)
#     else:
#         b1 = src_boxes.unsqueeze(1)
#         b2 = tgt_boxes.unsqueeze(0)

#     # 2. 提取参数 (保持不变)
#     cx1, cy1, w1, h1 = b1.unbind(-1)
#     cx2, cy2, w2, h2 = b2.unbind(-1)

#     # 3. 计算基础差值 (保持不变)
#     dx = cx1 - cx2
#     dy = cy1 - cy2

#     # 4. 计算 GCD 距离 (保持不变，这是核心逻辑)
#     # Part 1
#     c_dist1 = (dx / (w1 + eps))**2 + (dy / (h1 + eps))**2
#     wh_dist1 = (((w1 - w2) / (w1 + eps))**2 + ((h1 - h2) / (h1 + eps))**2) / 4.0
#     # Part 2
#     c_dist2 = (dx / (w2 + eps))**2 + (dy / (h2 + eps))**2
#     wh_dist2 = (((w1 - w2) / (w2 + eps))**2 + ((h1 - h2) / (h2 + eps))**2) / 4.0
#     # Average
#     gcd_2 = (c_dist1 + wh_dist1 + c_dist2 + wh_dist2) / 2.0

#     # ================= [关键修改点 START] =================
    
#     # [修改 1] 引入温度系数 tau，防止初期梯度消失
#     # 从零训练建议 tau=2.0 或 3.0
#     tau = 2.0  
#     gcd_similarity = torch.exp(-torch.sqrt(gcd_2 + eps) / tau)

#     # [修改 2] 角度连续化 (Sin/Cos)
#     # 解决 -pi 到 pi 的突变难训练问题
#     raw_angle = torch.atan2(dy, dx)
#     sin_angle = torch.sin(raw_angle)
#     cos_angle = torch.cos(raw_angle)

#     # [修改 3] Log 特征缩放
#     # 乘以 0.2 将范围从 0~5 压到 0~1 左右，避免在初期主导 MLP
#     scale_factor = 0.2
#     rel_x = torch.log(torch.abs(dx) / (w1 + eps) + 1.0) * scale_factor
#     rel_y = torch.log(torch.abs(dy) / (h1 + eps) + 1.0) * scale_factor
#     rel_w = torch.log((w1 + eps) / (w2 + eps)) * scale_factor
#     rel_h = torch.log((h1 + eps) / (h2 + eps)) * scale_factor

#     # ================= [关键修改点 END] =================

#     # 特征融合 (注意：现在是 7 维了！)
#     # 1(Sim) + 1(Sin) + 1(Cos) + 4(Rel) = 7
#     pos_embed = torch.cat([
#         gcd_similarity.unsqueeze(-1), 
#         sin_angle.unsqueeze(-1),
#         cos_angle.unsqueeze(-1),
#         rel_x.unsqueeze(-1), 
#         rel_y.unsqueeze(-1),
#         rel_w.unsqueeze(-1),
#         rel_h.unsqueeze(-1)
#     ], dim=-1) 
    
#     return pos_embed


# def gcd_relation_encoding_linear(src_boxes, tgt_boxes, eps=1e-7):
#     """
#     Formula B: Linear-Symmetric GCD Implementation
    
#     特点:
#     1. 完全对称: A->B 和 B->A 的特征互为相反数或相同。
#     2. 线性空间: 相对位置使用线性除法 (dx / joint_sigma)，而非 Log。
#     3. 数值风险: 远距离物体的特征值可能非常大 (>100)，可能导致梯度爆炸。
    
#     Args:
#         src_boxes (Tensor): [BS, N, 4]
#         tgt_boxes (Tensor): [BS, M, 4]
#     """
#     # 1. 维度对齐与广播
#     if src_boxes.dim() == 3:
#         b1 = src_boxes.unsqueeze(2)  
#         b2 = tgt_boxes.unsqueeze(1)
#     else:
#         b1 = src_boxes.unsqueeze(1)
#         b2 = tgt_boxes.unsqueeze(0)

#     # 2. 提取参数
#     cx1, cy1, w1, h1 = b1.unbind(-1)
#     cx2, cy2, w2, h2 = b2.unbind(-1)

#     # 3. 计算基础差值
#     dx = cx1 - cx2
#     dy = cy1 - cy2
    
#     # ================= [Formula B 核心逻辑] =================
    
#     # 4. 计算对称联合方差 (Symmetric Joint Variance)
#     # 这是 Formula B 的核心：分母融合了两个物体的尺寸
#     # var = w1^2 + w2^2
#     joint_var_w = w1.pow(2) + w2.pow(2) + eps
#     joint_var_h = h1.pow(2) + h2.pow(2) + eps
    
#     # 对应的联合标准差 (用于线性特征归一化)
#     # sigma = sqrt(w1^2 + w2^2)
#     joint_std_w = torch.sqrt(joint_var_w)
#     joint_std_h = torch.sqrt(joint_var_h)

#     # 5. 计算 GCD 综合距离 (Squared)
#     # 位置项: dx^2 / (w1^2 + w2^2)
#     term_loc = (dx.pow(2) / joint_var_w) + (dy.pow(2) / joint_var_h)
    
#     # 形状项: (w1-w2)^2 / (w1^2 + w2^2)
#     term_shape = ((w1 - w2).pow(2) / joint_var_w) + ((h1 - h2).pow(2) / joint_var_h)
    
#     # 总距离
#     gcd_dist = term_loc + term_shape

#     # ================= [特征向量构建] =================

#     # Feature 1: GCD 相似度 (0~1)
#     # 这一项是安全的，有 exp 压制
#     f_similarity = torch.exp(-torch.sqrt(gcd_dist))
    
#     # Feature 2: 相对角度 (-pi ~ pi)
#     # 这一项也是安全的
#     f_angle = torch.atan2(dy, dx)
    
#     # Feature 3 & 4: 对称线性相对坐标 (Symmetric Linear Relative Coords)
#     # [警告]!!! 这一项就是 Formula B 的风险源
#     # 如果两个小物体相距很远，这个值会非常大 (例如 1900 / 14 = 135.7)
#     f_rel_x = dx / joint_std_w
#     f_rel_y = dy / joint_std_h
    
#     # Feature 5 & 6: 相对宽高 (Log)
#     # 保持 Log 形式以维持反对称性 log(w1/w2)
#     f_rel_w = torch.log((w1 + eps) / (w2 + eps))
#     f_rel_h = torch.log((h1 + eps) / (h2 + eps))

#     # 6. 特征拼接
#     pos_embed = torch.cat([
#         f_similarity.unsqueeze(-1),
#         f_angle.unsqueeze(-1),
#         f_rel_x.unsqueeze(-1),
#         f_rel_y.unsqueeze(-1),
#         f_rel_w.unsqueeze(-1),
#         f_rel_h.unsqueeze(-1)
#     ], dim=-1) 
    
#     return pos_embed