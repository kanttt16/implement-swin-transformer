import torch
import torch.nn as nn
from .swin_transformer import SwinTransformer
from .upernet import UPerHead
from utils import resize


class UperNetSwin(nn.Module):
    """UperNet with Swin Transformer backbone for semantic segmentation."""
    
    def __init__(self, 
                 num_classes=150,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 pretrain_img_size=224,
                 in_channels=3,
                 patch_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 channels=512,
                 pool_scales=(1, 2, 3, 6),
                 align_corners=False):
        super(UperNetSwin, self).__init__()
        
        # Create Swin Transformer backbone
        self.backbone = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint
        )
        
        in_channels = [embed_dim * (2 ** i) for i in out_indices]
        
        # Create UPerNet head
        self.decode_head = UPerHead(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            align_corners=align_corners
        )
        
        self.align_corners = align_corners
        self.num_classes = num_classes
        
    def init_weights(self, pretrained_backbone=None):
        """Initialize the weights."""
        if pretrained_backbone is not None:
            self.backbone.init_weights(pretrained_backbone)
        else:
            # Initialize head weights
            for m in self.decode_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward function."""
        ori_h, ori_w = x.size()[2:]
        
        # Get multi-scale features from backbone
        features = self.backbone(x)
        
        # Forward through decode head
        output = self.decode_head(features)
        
        # Upsample to original size
        output = resize(
            output,
            size=(ori_h, ori_w),
            mode='bilinear',
            align_corners=self.align_corners
        )
        
        return output


def upernet_swin_tiny_patch4_window7_512(num_classes=150, pretrained_backbone=None, **kwargs):
    """Constructs a UperNet-Swin-Tiny model."""
    model = UperNetSwin(
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs
    )
    
    if pretrained_backbone is not None:
        model.init_weights(pretrained_backbone)
    
    return model