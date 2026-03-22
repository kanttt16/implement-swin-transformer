import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.upernet_swin import upernet_swin_tiny_patch4_window7_512
from datasets.ade20k import ADE20KDataset

# Try to import wandb
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def poly_lr_scheduler(current_iter, max_iter, initial_lr, power=0.9):
    return initial_lr * (1 - current_iter / max_iter) ** power


def worker_init_fn(worker_id):
    import random
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train UperNet with Swin Transformer on ADE20K')
    parser.add_argument('--data_root', type=str, default='ade20k/ADEChallengeData2016',
                        help='Path to ADE20K dataset')
    parser.add_argument('--pretrained', type=str, default='pre-weights/swin_tiny_patch4_window7_224.pth',
                        help='Path to pretrained Swin Transformer weights')
    parser.add_argument('--num_classes', type=int, default=150,
                        help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=160,
                        help='Number of training epochs (approx 160k iterations for batch size=4 matching original)')
    parser.add_argument('--lr', type=float, default=0.00006,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='Weight decay (original uses 0.02 for AdamW)')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Input crop size')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging even if available')
    parser.set_defaults(use_checkpoint=False)
    
    args = parser.parse_args()
    return args


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, print_freq, 
                   current_iter, total_iters, args, has_wandb=False, scaler=None):
    model.train()
    running_loss = 0.0
    num_samples = 0
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Update learning rate per iteration
        lr = poly_lr_scheduler(current_iter, total_iters, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass with autocast for mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)
        
        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / num_samples
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.6f}'})
        
        # Log to wandb
        if has_wandb:
            wandb.log({'train_loss': loss.item(), 'learning_rate': lr})
        
        current_iter += 1
    
    avg_loss = running_loss / num_samples
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} done in {epoch_time:.2f}s, average loss: {avg_loss:.4f}')
    
    return avg_loss, current_iter


def validate(model, dataloader, criterion, device, has_wandb=False, epoch=0, scaler=None):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    intersection = np.zeros(150)
    union = np.zeros(150)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            # Calculate mIoU
            preds = torch.argmax(outputs, dim=1)
            for pred, mask in zip(preds, masks):
                # Ignore 255
                valid_mask = mask != 255
                pred = pred[valid_mask]
                mask = mask[valid_mask]
                for cls in range(150):
                    pred_mask = (pred == cls)
                    true_mask = (mask == cls)
                    intersection[cls] += (pred_mask & true_mask).sum().item()
                    union[cls] += (pred_mask | true_mask).sum().item()
    
    avg_loss = running_loss / num_samples
    
    # Calculate mean IoU (only on classes that are present)
    iou = intersection / (union + 1e-8)
    # Only average over classes that are actually in the validation set
    # For ADE20K, all 150 classes should be present
    miou = np.mean(iou[iou > 0])
    
    # Log to wandb
    if has_wandb:
        wandb.log({'val_loss': avg_loss, 'val_mIoU': miou, 'epoch': epoch})
    
    print(f'Validation average loss: {avg_loss:.4f}, mIoU: {miou:.4f}')
    
    return avg_loss, miou


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = upernet_swin_tiny_patch4_window7_512(
        num_classes=args.num_classes,
        pretrained_backbone=args.pretrained,
        use_checkpoint=args.use_checkpoint
    )
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Data loading
    train_dataset = ADE20KDataset(
        root_dir=args.data_root,
        split='training',
        crop_size=(args.crop_size, args.crop_size)
    )
    val_dataset = ADE20KDataset(
        root_dir=args.data_root,
        split='validation',
        crop_size=(args.crop_size, args.crop_size)
    )
    
    # 在Windows上，num_workers>0会导致进程启动开销大，使用0可以解决问题
    # 如果需要多加载，可以使用persistent_workers=true来减少重启开销
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize wandb
    if has_wandb and not args.no_wandb:
        wandb.init(
            project='swin-transformer-segmentation',
            name='upernet_swin_tiny_ade20k',
            config=vars(args)
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Parameter grouping: no weight decay for position embedding & layer norm
    param_group_no_decay = []
    param_group_decay = []
    
    for name, param in model.named_parameters():
        if 'absolute_pos_embed' in name or 'relative_position_bias_table' in name or 'norm' in name:
            param_group_no_decay.append(param)
        else:
            param_group_decay.append(param)
    
    params = [
        {'params': param_group_decay, 'weight_decay': args.weight_decay},
        {'params': param_group_no_decay, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    # Initialize GradScaler for automatic mixed precision training
    # This speeds up training on NVIDIA GPUs with Tensor Cores
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Calculate total iterations matching original 160k iters
    total_iters = len(train_loader) * args.epochs
    current_iter = 0
    
    # Training loop
    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train one epoch
        _, current_iter = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, args.print_freq, current_iter, total_iters, 
            args, has_wandb and not args.no_wandb, scaler
        )
        
        # Validate
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_loss, miou = validate(
                model, val_loader, criterion, device, 
                has_wandb and not args.no_wandb, epoch, scaler
            )
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'best_miou': best_miou,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"New best model saved with mIoU: {best_miou:.4f}")
            
            # Log best mIoU
            if has_wandb and not args.no_wandb:
                wandb.log({'best_mIoU': best_miou, 'epoch': epoch})
        
        # Save checkpoint periodically
        if epoch % args.eval_freq == 0:
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    
    # Finish wandb
    if has_wandb and not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
