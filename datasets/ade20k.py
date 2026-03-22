import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RandomResize:
    """Random resize from (0.5, 2.0) of original shorter side."""
    def __init__(self, min_scale=0.5, max_scale=2.0, shorter_side=512):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.shorter_side = shorter_side
    
    def __call__(self, image, mask):
        w, h = image.size
        # Original shorter side
        if w < h:
            scale = self.shorter_side / w
        else:
            scale = self.shorter_side / h
        
        random_scale = random.uniform(self.min_scale, self.max_scale)
        scale = scale * random_scale
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = transforms.Resize((new_h, new_w))(image)
        mask = transforms.Resize((new_h, new_w), interpolation=Image.NEAREST)(mask)
        
        return image, mask


class RandomCrop:
    """Random crop to fixed size, pad if necessary."""
    def __init__(self, crop_size=(512, 512), cat_max_ratio=0.75, pad_val=0, seg_pad_val=255):
        self.crop_h, self.crop_w = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.pad_val = pad_val
        self.seg_pad_val = 255
    
    def __call__(self, image, mask):
        import torchvision.transforms.functional as F
        w, h = image.size
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Pad if necessary
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        
        if pad_h > 0 or pad_w > 0:
            # Pad image and mask
            image_pad = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), 
                             mode='constant', constant_values=self.pad_val)
            mask_pad = np.pad(mask_np, ((0, pad_h), (0, pad_w)), 
                            mode='constant', constant_values=self.seg_pad_val)
            # Convert back to PIL Image for torchvision operations
            image = Image.fromarray(image_pad)
            mask = Image.fromarray(mask_pad)
            h_pad, w_pad = image_pad.shape[:2]
        else:
            image_pad = image_np
            mask_pad = mask_np
            h_pad, w_pad = h, w
            image = Image.fromarray(image_np) if not isinstance(image, Image.Image) else image
            mask = Image.fromarray(mask_np) if not isinstance(mask, Image.Image) else mask
        
        # Try random crop until cat_max_ratio satisfied
        image_crop = None
        mask_crop = None
        for _ in range(10):
            # Random crop using torchvision
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_h, self.crop_w)
            )
            current_image = F.crop(image, i, j, h_crop, w_crop)
            current_mask = F.crop(mask, i, j, h_crop, w_crop)
            
            # Check category max ratio - convert once only when needed
            if self.cat_max_ratio >= 1.0:
                image_crop = current_image
                mask_crop = current_mask
                break
                
            current_mask_np = np.array(current_mask)
            unique, counts = np.unique(current_mask_np, return_counts=True)
            if 255 not in unique:
                image_crop = current_image
                mask_crop = current_mask
                break
            ratio = counts[unique == 255][0] / current_mask_np.size
            if ratio <= self.cat_max_ratio:
                image_crop = current_image
                mask_crop = current_mask
                break
        
        # If still not good enough after 10 tries, just take the last one
        if image_crop is None:
            # Give the last attempt result anyway
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_h, self.crop_w)
            )
            image_crop = F.crop(image, i, j, h_crop, w_crop)
            mask_crop = F.crop(mask, i, j, h_crop, w_crop)
        
        return image_crop, mask_crop


class PhotoMetricDistortion:
    """Photometric distortion from the original paper."""
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), 
                 saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
    
    def __call__(self, image):
        import torchvision.transforms.functional as F
        # Use torchvision transforms which are faster than numpy operations
        img = image
        
        # Random brightness
        if random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-self.brightness_delta/255.0, self.brightness_delta/255.0)
            img = F.adjust_brightness(img, brightness_factor)
        
        # Random contrast
        if random.random() < 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            img = F.adjust_contrast(img, contrast_factor)
        
        # Convert to HSV for saturation and hue adjustment
        # We still need to do this via numpy as torchvision doesn't have direct HSV adjustment
        img_np = np.array(img)
        hsv = np.array(Image.fromarray(img_np).convert('HSV')).astype(np.float32)
        
        # Random saturation
        if random.random() < 0.5:
            alpha = random.uniform(*self.saturation_range)
            hsv[:, :, 1] *= alpha
        
        # Random hue
        if random.random() < 0.5:
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            hsv[:, :, 0] += delta
            hsv[:, :, 0][hsv[:, :, 0] < 0] += 180
            hsv[:, :, 0][hsv[:, :, 0] > 180] -= 180
        
        # Back to RGB
        hsv = hsv.astype(np.uint8)
        return Image.fromarray(hsv).convert('RGB')


class ADE20KDataset(Dataset):
    """ADE20K Dataset.
    
    This matches the original data processing pipeline from Swin-Transformer-Semantic-Segmentation.
    """
    
    def __init__(self, root_dir, split='training', crop_size=(512, 512)):
        """
        Args:
            root_dir (string): Directory with ADE20K data.
            split (string): 'training' or 'validation'.
            crop_size: Input size after cropping.
        """
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        
        # Paths to images and annotations
        if split == 'training':
            self.img_dir = os.path.join(root_dir, 'images', 'training')
            self.ann_dir = os.path.join(root_dir, 'annotations', 'training')
        elif split == 'validation':
            self.img_dir = os.path.join(root_dir, 'images', 'validation')
            self.ann_dir = os.path.join(root_dir, 'annotations', 'validation')
        else:
            raise ValueError(f"Invalid split {split}, must be 'training' or 'validation'")
        
        # Get all image files
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.ann_files = sorted([f for f in os.listdir(self.ann_dir) if f.endswith('.png')])
        
        assert len(self.img_files) == len(self.ann_files), \
            f"Number of images ({len(self.img_files)}) != number of annotations ({len(self.ann_files)})"
        
        # Use original mean/std that matches pretraining
        # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375] -> converted to 0-1:
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32) / 255.0
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32) / 255.0
        
        # Data augmentation for training
        if self.split == 'training':
            self.random_resize = RandomResize(min_scale=0.5, max_scale=2.0, shorter_side=512)
            self.random_crop = RandomCrop(crop_size=crop_size)
            self.photo_metric = PhotoMetricDistortion()
        else:
            # Validation: resize shorter side to 512, then center crop
            self.resize_shorter = transforms.Resize(crop_size[0]) if crop_size[0] != crop_size[1] else transforms.Resize(crop_size)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        ann_path = os.path.join(self.ann_dir, self.ann_files[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(ann_path)
        
        # Convert ADE20K with reduce_zero_label=True: original 0 becomes ignore (255)
        # 1-150 -> 0-149 (total 150 classes)
        mask_np = np.array(mask)
        mask_np = mask_np - 1  # 1 -> 0, 150 -> 149, original 0 -> -1
        mask_np[mask_np == -1] = 255  # original background becomes ignore
        mask = Image.fromarray(mask_np)
        
        if self.split == 'training':
            # Random resize
            image, mask = self.random_resize(image, mask)
            # Random crop
            image, mask = self.random_crop(image, mask)
            # Random flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # Photometric distortion
            image = self.photo_metric(image)
        else:
            # Resize exactly to crop size for validation - ensures fixed dimensions for batch
            image = transforms.Resize(self.crop_size)(image)
            mask = transforms.Resize(self.crop_size, interpolation=Image.NEAREST)(mask)
        
        # Convert to tensor and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        return image_tensor, mask_tensor


if __name__ == '__main__':
    # Test the dataset
    dataset = ADE20KDataset(root_dir='../ade20k/ADEChallengeData2016', split='training')
    print(f"Dataset size: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Unique classes in mask: {torch.unique(mask)}")