import collections.abc
from einops import rearrange, repeat, reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from models.weight_init import trunc_normal_, constant_init_, kaiming_init_
import collections

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(
    attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w
):
    """
    Decomposed Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum("bythwc,hkc->bythwk", r_q, Rh)  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum("bythwc,wkc->bythwk", r_q, Rw)  # [B, H, q_t, qh, qw, k_w]

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    """
    Temporal Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
        torch.arange(q_t)[:, None] * q_t_ratio - torch.arange(k_t)[None, :] * k_t_ratio
    )
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        separate_qkv=False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.separate_qkv = separate_qkv
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first or separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)
        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, thw_shape):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"
            if not self.separate_qkv:
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, -1)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                q = k = v = x
                q = self.q(q).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
                k = self.k(k).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
                v = self.v(v).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None),
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None),
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None),
        )

        if self.pool_first:
            q_N = np.prod(q_shape) + 1 if self.has_cls_embed else np.prod(q_shape)
            k_N = np.prod(k_shape) + 1 if self.has_cls_embed else np.prod(k_shape)
            v_N = np.prod(v_shape) + 1 if self.has_cls_embed else np.prod(v_shape)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        layer_scale_init_value=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        dim_mul_in_att=False,
        separate_qkv=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
        )
        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim_out)),
                requires_grad=True,
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and np.prod(stride_skip) > 1
            else None
        )

    def forward(self, x, thw_shape=None):
        x_norm = self.norm1(x)
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)
        else:
            x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)
        if thw_shape:
            return x, thw_shape_new
        else:
            return x
		
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

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


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sine_cosine_pos_emb(n_position, d_hid): 
	''' Sinusoid position encoding table ''' 
	# TODO: make it with torch instead of numpy 
	def get_position_angle_vec(position): 
		return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

	sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

	return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DropPath(nn.Module):
	
	def __init__(self, dropout_p=None):
		super(DropPath, self).__init__()
		self.dropout_p = dropout_p

	def forward(self, x):
		return self.drop_path(x, self.dropout_p, self.training)
	
	def drop_path(self, x, dropout_p=0., training=False):
		if dropout_p == 0. or not training:
			return x
		keep_prob = 1 - dropout_p
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape).type_as(x)
		random_tensor.floor_()  # binarize
		output = x.div(keep_prob) * random_tensor
		return output

class ClassificationHead(nn.Module):
	"""Classification head for Video Transformer.
	
	Args:
		num_classes (int): Number of classes to be classified.
		in_channels (int): Number of channels in input feature.
		init_std (float): Std value for Initiation. Defaults to 0.02.
		kwargs (dict, optional): Any keyword argument to be used to initialize
			the head.
	"""

	def __init__(self,
				 num_classes,
				 in_channels,
				 init_std=0.02,
				 eval_metrics='finetune',
				 mode='default',
				 pretrain=False,
				 n_cells=None,
				 **kwargs):
		super().__init__()
		self.init_std = init_std
		self.eval_metrics = eval_metrics
		self.cls_head = nn.Linear(in_channels, num_classes)
		self.pretrain=pretrain
		self.num_classes=num_classes
		self.n_cells=n_cells
		self.mode=mode
		if mode=='swin': self.avgpool = nn.AdaptiveAvgPool3d(1)


		self.init_weights(self.cls_head)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			if self.eval_metrics == 'finetune':
				trunc_normal_(module.weight, std=self.init_std)
			else:
				module.weight.data.normal_(mean=0.0, std=0.01)
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		if self.mode=='swin':
			x = self.avgpool(x)
			x = torch.flatten(x, 1)

		if self.pretrain: x=rearrange(x,'(B N) H -> B (N H)',N=self.n_cells)
			
		cls_score = self.cls_head(x)
		
		return cls_score

class PatchEmbed(nn.Module):
	"""Images to Patch Embedding.

	Args:
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		tube_size (int): Size of temporal field of one 3D patch.
		in_channels (int): Channel num of input features. Defaults to 3.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
	"""

	def __init__(self,
				 img_size,
				 patch_size,
				 tube_size=2,
				 in_channels=3,
				 embed_dims=768,
				 conv_type='Conv2d'):
		super().__init__()
		self.img_size = _pair(img_size)
		self.patch_size = _pair(patch_size)

		num_patches = \
			(self.img_size[1] // self.patch_size[1])* \
			(self.img_size[0] // self.patch_size[0])
		assert (num_patches * self.patch_size[0] * self.patch_size[1] == 
			   self.img_size[0] * self.img_size[1],
			   'The image size H*W must be divisible by patch size')
		self.num_patches = num_patches

		# Use conv layer to embed
		if conv_type == 'Conv2d':
			self.projection = nn.Conv2d(
				in_channels,
				embed_dims,
				kernel_size=patch_size,
				stride=patch_size)
		elif conv_type == 'Conv3d':
			self.projection = nn.Conv3d(
				in_channels,
				embed_dims,
				kernel_size=(tube_size,patch_size,patch_size),
				stride=(tube_size,patch_size,patch_size))
		else:
			raise TypeError(f'Unsupported conv layer type {conv_type}')
			
		self.init_weights(self.projection)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		layer_type = type(self.projection)
		if layer_type == nn.Conv3d:
			x = rearrange(x, 'b t c h w -> b c t h w')
			x = self.projection(x)
			x = rearrange(x, 'b c t h w -> (b t) (h w) c')
		elif layer_type == nn.Conv2d:
			x = rearrange(x, 'b t c h w -> (b t) c h w')
			x = self.projection(x)
			x = rearrange(x, 'b c h w -> b (h w) c')
		else:
			raise TypeError(f'Unsupported conv layer type {layer_type}')
		
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
		return x, attn

class DividedTemporalAttentionWithPreNorm(nn.Module):
	"""Temporal Attention in Divided Space Time Attention. 
		A warp for torch.nn.MultiheadAttention.

	Args:
		embed_dims (int): Dimensions of embedding.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder.
		num_frames (int): Number of frames in the video.
		use_cls_token (bool): Whether to perform MSA on cls_token.
		attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
			0..
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Defaults to 0..
		layer_drop (dict): The layer_drop used when adding the shortcut.
			Defaults to `dict(type=DropPath, dropout_p=0.1)`.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
	"""

	def __init__(self,
				 embed_dims,
				 num_heads,
				 num_frames,
				 use_cls_token,
				 attn_drop=0.,
				 proj_drop=0.,
				 layer_drop=dict(type=DropPath, dropout_p=0.1),
				 norm_layer=nn.LayerNorm,
				 **kwargs):
		super().__init__()
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		self.num_frames = num_frames
		self.use_cls_token = use_cls_token

		self.norm = norm_layer(embed_dims)        
		#self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
		#								  **kwargs)
		self.attn = Attention(embed_dims, num_heads, qkv_bias=False, attn_drop=attn_drop) # batch first
										  
		self.proj_drop = nn.Dropout(proj_drop)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop= layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()
		if not use_cls_token:
			self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)
			self.init_weights(self.temporal_fc)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			constant_init_(module.weight, constant_value=0)
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, query, key=None, value=None, residual=None, return_attention=False, **kwargs):
		assert residual is None, (
			'Always adding the shortcut in the forward function')
		
		cls_token = query[:, 0, :].unsqueeze(1)
		if self.use_cls_token:
			residual = query
			query = query[:, 1:, :]
		else:
			query = query[:, 1:, :]
			residual = query

		b, n, d = query.size()
		p, t = n // self.num_frames, self.num_frames
		
		# Pre-Process
		query = rearrange(query, 'b (p t) d -> (b p) t d', p=p, t=t)
		if self.use_cls_token:
			cls_token = repeat(cls_token, 'b n d -> b (p n) d', p=p)
			cls_token = rearrange(cls_token, 'b p d -> (b p) 1 d')
			query = torch.cat((cls_token, query), 1)
		
		# Forward MSA
		query = self.norm(query)
		#query = rearrange(query, 'b n d -> n b d')
		#attn_out = self.attn(query, query, query)[0]
		#attn_out = rearrange(attn_out, 'n b d -> b n d')
		attn_out, attn_weights = self.attn(query)
		if return_attention:
			return attn_weights
		
		attn_out = self.layer_drop(self.proj_drop(attn_out.contiguous()))
		if not self.use_cls_token:
			attn_out = self.temporal_fc(attn_out)
		
		# Post-Process
		if self.use_cls_token:
			cls_token, attn_out = attn_out[:, 0, :], attn_out[:, 1:, :]
			cls_token = rearrange(cls_token, '(b p) d -> b p d', b=b)
			cls_token = reduce(cls_token, 'b p d -> b 1 d', 'mean')
			
			attn_out = rearrange(attn_out, '(b p) t d -> b (p t) d', p=p, t=t)
			attn_out = torch.cat((cls_token, attn_out), 1)
			new_query = residual + attn_out
		else:
			attn_out = rearrange(attn_out, '(b p) t d -> b (p t) d', p=p, t=t)
			new_query = residual + attn_out
			new_query = torch.cat((cls_token, new_query), 1)
		return new_query


class DividedSpatialAttentionWithPreNorm(nn.Module):
	"""Spatial Attention in Divided Space Time Attention.
		A warp for torch.nn.MultiheadAttention.
		
	Args:
		embed_dims (int): Dimensions of embedding.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder.
		num_frames (int): Number of frames in the video.
		use_cls_token (bool): Whether to perform MSA on cls_token.
		attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
			0..
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Defaults to 0..
		layer_drop (dict): The layer_drop used when adding the shortcut.
			Defaults to `dict(type=DropPath, dropout_p=0.1)`.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
	"""

	def __init__(self,
				 embed_dims,
				 num_heads,
				 num_frames,
				 use_cls_token,
				 attn_drop=0.,
				 proj_drop=0.,
				 layer_drop=dict(type=DropPath, dropout_p=0.1),
				 norm_layer=nn.LayerNorm,
				 **kwargs):
		super().__init__()
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		self.num_frames = num_frames
		self.use_cls_token = use_cls_token
		
		self.norm = norm_layer(embed_dims)
		#self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
		#								  **kwargs)
		self.attn = Attention(embed_dims, num_heads, qkv_bias=False, attn_drop=attn_drop) # batch first
										  
		self.proj_drop = nn.Dropout(proj_drop)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop= layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

		self.init_weights()

	def init_weights(self):
		pass

	def forward(self, query, key=None, value=None, residual=None, return_attention=False, **kwargs):
		assert residual is None, (
			'Always adding the shortcut in the forward function')
		
		cls_token = query[:, 0, :].unsqueeze(1)
		if self.use_cls_token:
			residual = query
			query = query[:, 1:, :]


		else:
			query = query[:, 1:, :]
			residual = query

		b, n, d = query.size()
		p, t = n // self.num_frames, self.num_frames
		
		# Pre-Process
		query = rearrange(query, 'b (p t) d -> (b t) p d', p=p, t=t)
		if self.use_cls_token:
			cls_token = repeat(cls_token, 'b n d -> b (t n) d', t=t)
			cls_token = rearrange(cls_token, 'b t d -> (b t) 1 d')
			query = torch.cat((cls_token, query), 1)
		
		# Forward MSA
		query = self.norm(query)
		#query = rearrange(query, 'b n d -> n b d')
		#attn_out = self.attn(query, query, query)[0]
		#attn_out = rearrange(attn_out, 'n b d -> b n d')
		attn_out, attn_weights = self.attn(query)
		if return_attention:
			return attn_weights

		attn_out = self.layer_drop(self.proj_drop(attn_out.contiguous()))
		
		# Post-Process
		if self.use_cls_token:
			cls_token, attn_out = attn_out[:, 0, :], attn_out[:, 1:, :]
			cls_token = rearrange(cls_token, '(b t) d -> b t d', b=b)
			cls_token = reduce(cls_token, 'b t d -> b 1 d', 'mean')
			
			attn_out = rearrange(attn_out, '(b t) p d -> b (p t) d', p=p, t=t)
			attn_out = torch.cat((cls_token, attn_out), 1)
			new_query = residual + attn_out
		else:
			attn_out = rearrange(attn_out, '(b t) p d -> b (p t) d', p=p, t=t)
			new_query = residual + attn_out
			new_query = torch.cat((cls_token, new_query), 1)
		return new_query


class MultiheadAttentionWithPreNorm(nn.Module):
	"""Implements MultiheadAttention with residual connection.
	
	Args:
		embed_dims (int): The embedding dimension.
		num_heads (int): Parallel attention heads.
		attn_drop (float): A Dropout layer on attn_output_weights.
			Default: 0.0.
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Default: 0.0.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
		layer_drop (obj:`ConfigDict`): The layer_drop used
			when adding the shortcut.
		batch_first (bool): When it is True,  Key, Query and Value are shape of
			(batch, n, embed_dim), otherwise (n, batch, embed_dim).
			 Default to False.
	"""

	def __init__(self,
				 embed_dims,
				 num_heads,
				 attn_drop=0.,
				 proj_drop=0.,
				 norm_layer=nn.LayerNorm,
				 layer_drop=dict(type=DropPath, dropout_p=0.),
				 batch_first=False,
				 **kwargs):
		super().__init__()
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		#self.batch_first = batch_first
		
		self.norm = norm_layer(embed_dims)
		#self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
		#								  **kwargs)
		self.attn = Attention(embed_dims, num_heads, qkv_bias=False, attn_drop=attn_drop) # batch first

		self.proj_drop = nn.Dropout(proj_drop)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop= layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

	def forward(self,
				query,
				key=None,
				value=None,
				residual=None,
				attn_mask=None,
				key_padding_mask=None,
				return_attention=False,
				**kwargs):
		residual = query
		
		query = self.norm(query)
		#if self.batch_first:
		#	query = query.transpose(0, 1)
		#attn_out = self.attn(
		#	query=query,
		#	key=query,
		#	value=query,
		#	attn_mask=attn_mask,
		#	key_padding_mask=key_padding_mask)[0]
		#attn_out = self.attn(query, query, query)[0]
		#if self.batch_first:
		#	attn_out = attn_out.transpose(0, 1)
		attn_out, attn_weights = self.attn(query)
		if return_attention:
			return attn_weights

		new_query = residual + self.layer_drop(self.proj_drop(attn_out))
		return new_query


class FFNWithPreNorm(nn.Module):
	"""Implements feed-forward networks (FFNs) with residual connection.
	
	Args:
		embed_dims (int): The feature dimension. Same as
			`MultiheadAttention`. Defaults: 256.
		hidden_channels (int): The hidden dimension of FFNs.
			Defaults: 1024.
		num_layers (int, optional): The number of fully-connected layers in
			FFNs. Default: 2.
		act_layer (dict, optional): The activation layer for FFNs.
			Default: nn.GELU
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
		dropout_p (float, optional): Probability of an element to be
			zeroed in FFN. Default 0.0.
		layer_drop (obj:`ConfigDict`): The layer_drop used
			when adding the shortcut.
	"""
	
	def __init__(self,
				 embed_dims=256,
				 hidden_channels=1024,
				 num_layers=2,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 dropout_p=0.,
				 layer_drop=None,
				 **kwargs):
		super().__init__()
		assert num_layers >= 2, 'num_layers should be no less ' \
			f'than 2. got {num_layers}.'
		self.embed_dims = embed_dims
		self.hidden_channels = hidden_channels
		self.num_layers = num_layers
		
		self.norm = norm_layer(embed_dims)
		layers = []
		in_channels = embed_dims
		for _ in range(num_layers - 1):
			layers.append(
				nn.Sequential(
					nn.Linear(in_channels, hidden_channels),
					act_layer(),
					nn.Dropout(dropout_p)))
			in_channels = hidden_channels
		layers.append(nn.Linear(hidden_channels, embed_dims))
		layers.append(nn.Dropout(dropout_p))
		self.layers = nn.ModuleList(layers)
		
		if layer_drop:
			dropout_p = layer_drop.pop('dropout_p')
			layer_drop= layer_drop.pop('type')
			self.layer_drop = layer_drop(dropout_p)  
		else:
			self.layer_drop = nn.Identity()

	def forward(self, x):
		residual = x
		
		x = self.norm(x)
		for layer in self.layers:
			x = layer(x)
			
		return residual + self.layer_drop(x)


class TransformerContainer(nn.Module):

	def __init__(self, 
				 num_transformer_layers,
				 embed_dims,
				 num_heads,
				 num_frames,
				 hidden_channels,
				 operator_order,
				 drop_path_rate=0.1,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 num_layers=2):
		super().__init__()
		self.layers = nn.ModuleList([])
		self.num_transformer_layers = num_transformer_layers
		
		dpr = np.linspace(0, drop_path_rate, num_transformer_layers)
		for i in range(num_transformer_layers):	
			self.layers.append(
				BasicTransformerBlock(
					embed_dims=embed_dims,
					num_heads=num_heads,
					num_frames=num_frames,
					hidden_channels=hidden_channels,
					operator_order=operator_order,
					norm_layer=norm_layer,
					act_layer=act_layer,
					num_layers=num_layers,
					dpr=dpr[i]))
		
	def forward(self, x, return_attention=False):
		layer_idx = 0
		for layer in self.layers:
			if layer_idx >= self.num_transformer_layers-1 and return_attention:
				x = layer(x, return_attention=True)
			else:
				x = layer(x)
			layer_idx += 1
		return x

class BasicTransformerBlock(nn.Module):

	def __init__(self, 
				 embed_dims,
				 num_heads,
				 num_frames,
				 hidden_channels,
				 operator_order,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 num_layers=2,
				 dpr=0,
				 ):

		super().__init__()
		self.attentions = nn.ModuleList([])
		self.ffns = nn.ModuleList([])
		
		for i, operator in enumerate(operator_order):
			if operator == 'self_attn':
				self.attentions.append(
					MultiheadAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						batch_first=True,
						norm_layer=nn.LayerNorm,
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			elif operator == 'time_attn':
				self.attentions.append(
					DividedTemporalAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						num_frames=num_frames,
						norm_layer=norm_layer,
						use_cls_token=(i==len(operator_order)-2),
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			elif operator == 'space_attn':
				self.attentions.append(
					DividedSpatialAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						num_frames=num_frames,
						norm_layer=norm_layer,
						use_cls_token=(i==len(operator_order)-2),
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			elif operator == 'ffn':
				self.ffns.append(
					FFNWithPreNorm(
						embed_dims=embed_dims,
						hidden_channels=hidden_channels,
						num_layers=num_layers,
						act_layer=act_layer,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			else:
				raise TypeError(f'Unsupported operator type {operator}')
		
	def forward(self, x, return_attention=False):
		attention_idx = 0
		for layer in self.attentions:
			if attention_idx >= len(self.attentions)-1 and return_attention:
				x = layer(x, return_attention=True)
				return x
			else:x = layer(x)
			attention_idx += 1
		for layer in self.ffns:
			x = layer(x)
		return x
