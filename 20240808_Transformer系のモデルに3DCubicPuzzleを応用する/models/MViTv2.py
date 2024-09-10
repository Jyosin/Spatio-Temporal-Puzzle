# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_component import MultiScaleBlock,get_3d_sincos_pos_embed
from torch.nn.init import trunc_normal_
from utils.utils import print_on_rank_zero

def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        print(f"min width {min_width}")
        print(f"width {width} divisor {divisor}")
        print(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

def calc_mvit_feature_geometry(
        num_frames,
        patch_stride,
        train_crop_size,
        depth,
        pool_q_stride
):
    feat_size = [
        [
            (
                num_frames // patch_stride[0]
                if len(patch_stride) > 2
                else 1
            ),
            train_crop_size // patch_stride[-2],
            train_crop_size // patch_stride[-1],
        ]
        for i in range(depth)
    ]
    feat_stride = [
        [
            patch_stride[0] if len(patch_stride) > 2 else 1,
            patch_stride[-2],
            patch_stride[-1],
        ]
        for i in range(depth)
    ]
    for _, x in enumerate(pool_q_stride):
        for i in range(depth):
            if i >= x[0]:
                for j in range(len(feat_size[i])):
                    feat_size[i][j] = feat_size[i][j] // x[j + 1]
                    feat_stride[i][j] = feat_stride[i][j] * x[j + 1]
    return feat_size, feat_stride

class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, keep_spatial=False):
        x = self.proj(x)
        if keep_spatial:
            return x, x.shape
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2), x.shape

class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(
                self, 
                pretrained=None,
                train_crop_size=224,
                test_crop_size=224,
                n_frames=32,
                input_channel_num=[3],
                mode="conv",
                pool_first=False,
                cls_embed_on=True,
                patch_kernel=[3,7,7],
                patch_stride=[2,4,4],
                patch_padding=[1,3,3],
                patch_2d=False,
                embed_dim=96,
                num_heads=1,
                mlp_ratio=4.0,
                qkv_bias=True,
                droppath_rate=0.1,
                layer_scale_init_value=0.0,
                depth=24,
                norm="layernorm",
                dim_mul= [[2, 2.0], [5, 2.0], [21, 2.0]],
                head_mul= [[2, 2.0], [5, 2.0], [21, 2.0]],
                pool_kv_stride={},
                pool_kv_stride_adaptive=  [1, 8, 8], # githubのパラメータ：[1,8,8]　事前学習用：[1,2,2]
                pool_q_stride=  [[2,1, 2, 2], [5, 1, 2, 2],  [21, 1, 2, 2]],
                pool_kvq_kernel=None,
                zero_decay_pos_cls=False,
                norm_stem=False,
                sep_pos_embed=True,
                dropout_rate=0.0,
                use_abs_pos=True,
                rel_pos_spatial=False,
                rel_pos_temporal=False,
                rel_pos_zero_init=False,
                residual_pooling=False,
                dim_mul_in_att=False, # githubのパラメータ：False
                separate_qkv=False,
                head_init_scale=1.0,
                use_mean_pooling=False,
                use_fixed_sincos_pos=False
                 ):
        super().__init__()
        self.pretrained=pretrained
        # Get parameters.
        assert train_crop_size == test_crop_size
        pool_first = pool_first
        # prepare input.
        spatial_size = train_crop_size
        temporal_size = n_frames
        in_chans = input_channel_num[0]
        self.use_2d_patch = patch_2d
        self.patch_stride = patch_stride
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.t = n_frames // self.patch_stride[0]
        self.h = train_crop_size // self.patch_stride[1]
        self.w = train_crop_size // self.patch_stride[2]
        # prepare output.
        self.embed_dim = embed_dim
        # prepare backbone
        num_heads = num_heads
        mlp_ratio = mlp_ratio
        qkv_bias = qkv_bias
        self.drop_rate = dropout_rate
        depth = depth
        drop_path_rate = droppath_rate
        layer_scale_init_value = layer_scale_init_value
        head_init_scale = head_init_scale
        mode = mode
        self.cls_embed_on = cls_embed_on
        self.use_mean_pooling = use_mean_pooling
        # params for positional embedding
        self.use_abs_pos = use_abs_pos
        self.use_fixed_sincos_pos = use_fixed_sincos_pos
        self.sep_pos_embed = sep_pos_embed
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        self.zero_decay_pos_cls = zero_decay_pos_cls
        self.patch_kernel=patch_kernel
        if norm == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            conv_2d=self.use_2d_patch,
        )

        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim)
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        _dim_mul, _head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(dim_mul)):
            _dim_mul[dim_mul[i][0]] = dim_mul[i][1]
        for i in range(len(head_mul)):
            _head_mul[head_mul[i][0]] = head_mul[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        for i in range(len(pool_q_stride)):
            stride_q[pool_q_stride[i][0]] = pool_q_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride[i][1:]
                ]

        # if pool_kv_stride_adaptive is not none, initialize pool_kv_stride.
        if pool_kv_stride_adaptive is not None:
            _stride_kv = pool_kv_stride_adaptive
            pool_kv_stride = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append([i] + _stride_kv)

        for i in range(len(pool_kv_stride)):
            stride_kv[pool_kv_stride[i][0]] = pool_kv_stride[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_kv_stride[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        
        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        input_size = self.patch_dims


        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, _head_mul[i])
            if dim_mul_in_att:
                dim_out = round_width(
                    embed_dim,
                    _dim_mul[i],
                    divisor=round_width(num_heads, _head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    _dim_mul[i + 1],
                    divisor=round_width(num_heads, _head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=rel_pos_zero_init,
                residual_pooling=residual_pooling,
                dim_mul_in_att=dim_mul_in_att,
                separate_qkv=separate_qkv,
            )

            self.blocks.append(attention_block)
            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.init_weights()

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(
            num_frames=n_frames,
            patch_stride=patch_stride,
            train_crop_size=train_crop_size,
            depth=depth,
            pool_q_stride=pool_q_stride
        )

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model_state']
        
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_kernel[0],1,1) / self.patch_kernel[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "attn.pool" in k]
        for k in relative_position_bias_table_keys:
            state_dict[k] = state_dict[k].unsqueeze(2)

        missing_keys,unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print_on_rank_zero(f'missing_keys:{missing_keys}\n '
						   f'unexpected_keys:{unexpected_keys}')
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self):
        if self.pretrained is not None: 
            try:
                self.inflate_weights()
            except:
                print("fail to load pretrain model")
        else: self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :].reshape(1, p_t, p_h, p_w, -1).permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False,use_pos_embed=True):
        # B T C H W → B C T H W
        x=x.transpose(1,2)
        
        #x = x[0]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.t))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        
        assert len(bcthw) == 5 and (T, H, W) == (self.t, self.h, self.w), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if use_pos_embed and self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if use_pos_embed and self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        for blk in self.blocks: x, thw = blk(x, thw)

        if self.use_mean_pooling:
            if self.cls_embed_on:
                x = x[:, 1:]
            x = x.mean(1)
            x = self.norm(x)
        elif self.cls_embed_on:
            x = self.norm(x)
            x = x[:, 0]
        else:  # this is default, [norm->mean]
            x = self.norm(x)
            x = x.mean(1)

        return x