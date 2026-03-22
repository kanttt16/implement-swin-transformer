import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This is a simplified replacement for mmcv.cnn.ConvModule.
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=None, 
                 dilation=1, 
                 groups=1, 
                 bias=False,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None,  # For compatibility with original code
                 inplace=True,  # For compatibility
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()

        # Automatically calculate padding if not provided
        if padding is None:
            if isinstance(kernel_size, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                # Handle tuple kernel_size
                padding = tuple((k - 1) // 2 * d for k, d in zip(kernel_size, (dilation, dilation) if isinstance(dilation, int) else dilation))
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        # Add normalization if norm_cfg is provided
        if norm_cfg is not None:
            if norm_cfg['type'] == 'BN':
                self.bn = nn.BatchNorm2d(out_channels)
            else:
                raise NotImplementedError(f"Normalization type {norm_cfg['type']} not implemented")
        else:
            self.bn = None

        # Add activation if act_cfg is provided
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.activate = nn.ReLU(inplace=False)
            elif act_cfg['type'] == 'GELU':
                self.activate = nn.GELU()
            else:
                raise NotImplementedError(f"Activation type {act_cfg['type']} not implemented")
        else:
            self.activate = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x