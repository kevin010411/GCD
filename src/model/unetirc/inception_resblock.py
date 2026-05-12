import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from .blocks.inceptionnext_v2 import InceptionNeXtBlock_V2
from .blocks.utils import LayerNorm
from .blocks.cbam import CBAM
from src.utils import MODEL


@MODEL.register_module()
class INCEPTION_RESBLOCK(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        patch_size=2,
        kernel_size=7,
        exp_rate=4,
        feature_size=48,
        depths=[3, 3, 9, 3],
        drop_path_rate=0.0,
        use_init_weights=False,
        is_conv_stem=False,
        skip_encoder_name=None,
        deep_sup=False,
        first_feature_size_half=False,
        res_block=True,
        **kwargs,
    ) -> None:
        super().__init__()

        feature_sizes = [feature_size * (2**i) for i in range(len(depths))]
        first_feature_size = (
            feature_sizes[0] // 2 if first_feature_size_half else feature_sizes[0]
        )

        decoder_norm_name = "instance"
        spatial_dims = 3

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=first_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=decoder_norm_name,
            res_block=not res_block,
        )

        self.backbone = Backbone(
            in_channels=in_channels,
            patch_size=patch_size,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            feature_sizes=feature_sizes,
            depths=depths,
            drop_path_rate=drop_path_rate,
            use_init_weights=use_init_weights,
            is_conv_stem=is_conv_stem,
        )

        # Skip Connections
        self.skip_encoder_name = skip_encoder_name
        if self.skip_encoder_name == "cbam":
            self.skip_encoder0 = nn.Identity()
            self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
            self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
            self.skip_encoder3 = CBAM(feature_sizes[2], reduction=16, kernel_size=7)
            self.skip_encoder4 = CBAM(feature_sizes[3], reduction=16, kernel_size=7)

        elif self.skip_encoder_name == "hybrid":
            # Level 0 (Stem)
            self.skip_encoder0 = nn.Identity()
            # Level 1 (Stage 0): CBAM
            self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
            # Level 2 (Stage 1): CBAM
            self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
            # Level 3 (Stage 2): Identity
            self.skip_encoder3 = nn.Identity()
            # Level 4 (Stage 3): Identity
            self.skip_encoder4 = nn.Identity()

        self.bottleneck = nn.Sequential(
            LayerNorm(feature_sizes[3], eps=1e-6, data_format="channels_first"),
            nn.Conv3d(feature_sizes[3], feature_sizes[3] * 2, kernel_size=2, stride=2),
            CBAM(feature_sizes[3] * 2, reduction=16, kernel_size=7),
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3] * 2,
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=first_feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.out_block = UnetOutBlock(
            spatial_dims=3, in_channels=first_feature_size, out_channels=out_channels
        )

        self.deep_sup = deep_sup
        if deep_sup:
            self.ds_block1 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[0],
                out_channels=out_channels,
            )
            self.ds_block2 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[1],
                out_channels=out_channels,
            )

    def forward(self, x):
        # 輔助函式，用於處理多輸入的 checkpoint
        def forward_multiple(x):
            module, in1, in2 = x
            return module(in1, in2)

        # 使用 Checkpoint 以節省 VRAM
        enc0 = checkpoint(self.encoder0, x, use_reentrant=False)
        hidden_states_out = checkpoint(self.backbone, x, use_reentrant=False)
        enc1, enc2, enc3, enc4 = hidden_states_out

        bn = checkpoint(self.bottleneck, enc4, use_reentrant=False)

        dec5 = checkpoint(
            forward_multiple, (self.decoder5, bn, enc4), use_reentrant=False
        )
        dec4 = checkpoint(
            forward_multiple, (self.decoder4, dec5, enc3), use_reentrant=False
        )
        dec3 = checkpoint(
            forward_multiple, (self.decoder3, dec4, enc2), use_reentrant=False
        )
        dec2 = checkpoint(
            forward_multiple, (self.decoder2, dec3, enc1), use_reentrant=False
        )
        dec1 = checkpoint(
            forward_multiple, (self.decoder1, dec2, enc0), use_reentrant=False
        )

        # Grad-CAM 梯度保留邏輯
        if x.requires_grad:
            self.layers = {
                "dec1": dec1,
                "dec2": dec2,
                "dec3": dec3,
                "dec4": dec4,
                "dec5": dec5,
                "enc4": enc4,
                "enc3": enc3,
                "enc2": enc2,
                "enc1": enc1,
                "enc0": enc0,
            }
            for v in self.layers.values():
                v.retain_grad()

        out = self.out_block(dec1)

        if self.deep_sup and self.training:
            out1 = self.ds_block1(dec2)
            out2 = self.ds_block2(dec3)
            return [out, out1, out2]
        else:
            return out


class Backbone(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        kernel_size,
        exp_rate,
        feature_sizes,
        depths,
        drop_path_rate,
        use_init_weights,
        is_conv_stem,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        if is_conv_stem:
            stem = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    feature_sizes[0],
                    kernel_size=7,
                    stride=patch_size,
                    padding=3,
                ),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first"),
            )
        else:
            stem = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    feature_sizes[0],
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first"),
            )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(feature_sizes[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(
                    feature_sizes[i], feature_sizes[i + 1], kernel_size=2, stride=2
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[
                    InceptionNeXtBlock_V2(
                        dim=feature_sizes[i],
                        hwd_kernel_size=kernel_size,
                        exp_rate=exp_rate,
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        if use_init_weights:
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
