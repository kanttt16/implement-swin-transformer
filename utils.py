import torch
import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode='bilinear', align_corners=None):
    """Resize input to given size or scale factor.
    
    This is a wrapper around F.interpolate for compatibility.
    """
    return F.interpolate(input, size=size, scale_factor=scale_factor, 
                         mode=mode, align_corners=align_corners)