import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torchmetrics.segmentation import DiceScore
import segmentation_models_pytorch as smp

from monai.losses import GeneralizedDiceFocalLoss
from schedulefree import RAdamScheduleFree

import os

from collections import OrderedDict
import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_env():
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

set_env()

class CFG:
    backbone = "tu-convnext_base"
    save_syntax = "upernet"
    autocast = True
    image_size=512
    fold = 4
    debug = True
    
    num_epochs = 100
    mask_num=13
    batch_size = 16
    learning_rate = 1e-4
    train_augmentation = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                border_mode=0, rotate_limit=90, scale_limit=0.5, shift_limit=0.2
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.FancyPCA(p=0.5),
            A.OneOf(
                [
                    A.RandomFog(fog_coef_range=(0.3, 0.5)),
                    A.RandomSunFlare(),
                    A.RandomShadow()
                ], p=0.2
            ),
            A.MotionBlur(blur_limit=(3, 15), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CoarseDropout(max_holes=10, max_height=100, max_width=100, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )

    valid_augmentation = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Flag for whether to use multi-GPU
    debug=False

    model_path = f"models/upernet/fold{fold}.pth"
    LABEL2CH = {
        0: 0,  50: 0, 255: 0, 
        5: 1,  11: 2, 12: 3, 13: 4,
        21: 5,  22: 6, 23: 7, 24: 8,
        25: 9,  31:10, 32:11, 33:12
    }

class CholecSeg8kDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mask_path  = self.df['file'][idx]
        image_path = mask_path.replace('_watershed_mask', '')
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        idx_mask = np.vectorize(CFG.LABEL2CH.get, otypes=[np.uint8])(mask_raw)
        # Pass a 2D class index mask to Albumentations (one-hot is created downstream as needed)
        mask_idx = idx_mask.astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask_idx)
            image = out["image"]  # torch.Tensor [C,H,W]
            mask_idx = out["mask"]  # torch.Tensor [H,W] or np.ndarray
        else:
            # Ensure minimal tensorization even without transforms (normally use valid_augmentation for val/test consistency)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_idx = torch.from_numpy(mask_idx).long()

        if not torch.is_tensor(mask_idx):
            mask_idx = torch.from_numpy(np.asarray(mask_idx)).long()
        else:
            mask_idx = mask_idx.long()

        return image, mask_idx


def _denorm_rgb_image(img_chw: torch.Tensor) -> np.ndarray:
    """img_chw: float tensor [3,H,W] normalized by Imagenet mean/std -> uint8 RGB HWC"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device).view(3, 1, 1)
    x = (img_chw * std + mean).clamp(0, 1)
    x = (x * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return x


def save_debug_batch(split: str, images: torch.Tensor, mask_idx: torch.Tensor, out_dir: str, max_items: int = 4) -> None:
    """
    Save the first batch input image and segmentation mask.
    - image: denormalize and save as PNG
    - mask: save class index (0..12) as-is in PNG
    """
    os.makedirs(out_dir, exist_ok=True)
    b = int(images.shape[0])
    n = min(b, max_items)

    # Fixed palette for 13 classes (BGR)
    palette_bgr = np.array(
        [
            [0, 0, 0],        # 0 background
            [0, 0, 255],      # 1
            [0, 255, 0],      # 2
            [255, 0, 0],      # 3
            [0, 255, 255],    # 4
            [255, 0, 255],    # 5
            [255, 255, 0],    # 6
            [128, 0, 0],      # 7
            [0, 128, 0],      # 8
            [0, 0, 128],      # 9
            [128, 128, 0],    # 10
            [128, 0, 128],    # 11
            [0, 128, 128],    # 12
        ],
        dtype=np.uint8,
    )

    for i in range(n):
        img_rgb = _denorm_rgb_image(images[i])
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        m = mask_idx[i].detach().cpu().numpy().astype(np.uint8)
        m = np.clip(m, 0, CFG.mask_num - 1).astype(np.uint8)
        mask_color = palette_bgr[m]  # (H,W,3)
        overlay = cv2.addWeighted(img_bgr, 0.7, mask_color, 0.3, 0.0)

        img_path = os.path.join(out_dir, f"{split}_img_{i:02d}.png")
        mask_path = os.path.join(out_dir, f"{split}_mask_{i:02d}.png")
        mask_color_path = os.path.join(out_dir, f"{split}_mask_color_{i:02d}.png")
        overlay_path = os.path.join(out_dir, f"{split}_overlay_{i:02d}.png")

        cv2.imwrite(img_path, img_bgr)
        cv2.imwrite(mask_path, m)
        cv2.imwrite(mask_color_path, mask_color)
        cv2.imwrite(overlay_path, overlay)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    optimizer.train()
    epoch_loss = 0
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast and device.type == "cuda")
    
    with tqdm(loader, desc="Training") as pbar:
        for images, mask_idx in pbar:
            images = images.to(device, non_blocking=True)
            mask_idx = mask_idx.to(device, non_blocking=True).long()
            masks = F.one_hot(mask_idx, num_classes=CFG.mask_num).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=CFG.autocast and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return epoch_loss / len(loader)

def validate(model, loader, train_criterion, criterion, device, epoch, optimizer=None):
    if optimizer is not None:
        optimizer.eval()
    model.eval()
    criterion.reset()
    
    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for images, mask_idx in pbar:
                images = images.to(device, non_blocking=True)
                mask_idx = mask_idx.to(device, non_blocking=True).long()
                
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=CFG.autocast and device.type == "cuda"):
                    logits = model(images)
                    pred_idx = logits.argmax(dim=1)

                preds_1h = F.one_hot(pred_idx, num_classes=CFG.mask_num).permute(0, 3, 1, 2).int()
                target_1h = F.one_hot(mask_idx, num_classes=CFG.mask_num).permute(0, 3, 1, 2).int()
                criterion.update(preds_1h.cpu(), target_1h.cpu())
        
        val_loss = criterion.compute()
        for i, scr in enumerate(val_loss):
            print(f"indice {i} | {float(scr):.4f}")
    return val_loss.mean().item()

def test(model, loader, criterion, device):
    model.eval()
    val_loss = []

    criterion.to(device)
    criterion.reset()

    with torch.no_grad():
        with tqdm(loader, desc="Validation") as pbar:
            for i, batch in enumerate(pbar):
                # Compatibility: support both (image, mask) and (image, mask, video_start, flag) formats
                if len(batch) == 2:
                    images, masks = batch
                    video_start, flag = None, None
                elif len(batch) == 4:
                    images, masks, video_start, flag = batch
                else:
                    raise ValueError(f"Unexpected batch format: len={len(batch)}")

                images = images.to(device, non_blocking=True)
                masks = masks.long().to(device, non_blocking=True)

                logits = model(images)
                pred_idx = logits.argmax(dim=1)
                preds_1h = F.one_hot(pred_idx, num_classes=CFG.mask_num).permute(0, 3, 1, 2).int()
                target_1h = F.one_hot(masks, num_classes=CFG.mask_num).permute(0, 3, 1, 2).int()

                # Average per video only when video boundary information is available
                if video_start is not None and bool(video_start[0]) and i > 0:
                    val_loss.append(criterion.compute().cpu())
                    criterion.reset()

                if flag is None or bool(flag[0]):
                    criterion.update(preds_1h, target_1h)

    val_loss.append(criterion.compute().cpu())
    val_loss = np.array(val_loss).mean(0)
    for i, scr in enumerate(val_loss):
        print(f"indice {i} | {float(scr):.4f}")
    return val_loss.mean().item()

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        elif k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v
    return new_state_dict

def main():
    device = CFG.device
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    torch.manual_seed(42)
    np.random.seed(42)
    

    model = smp.UPerNet(
        encoder_name=CFG.backbone, encoder_weights="imagenet", classes=13, activation="softmax"
    )
    model = model.to(device)
    train_csv = os.path.join("splits", f"cholecseg8k_train_{CFG.fold}.csv")
    valid_csv = os.path.join("splits", f"cholecseg8k_test_{CFG.fold}.csv")
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    train_dataset = CholecSeg8kDataset(train_df, transform=CFG.train_augmentation)
    val_dataset = CholecSeg8kDataset(valid_df, transform=CFG.valid_augmentation)
    test_dataset = CholecSeg8kDataset(valid_df, transform=CFG.valid_augmentation)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )

    if CFG.debug:
        debug_dir = os.path.join(base_dir, "debug_outputs")
        # First 1 batch from training set
        images, masks = next(iter(train_loader))
        save_debug_batch("train", images, masks, os.path.join(debug_dir, "train"), max_items=4)

        # First 1 batch from test set
        images, masks = next(iter(test_loader))
        save_debug_batch("test", images, masks, os.path.join(debug_dir, "test"), max_items=4)

        print(f"[DEBUG] Saved debug outputs to: {debug_dir}")
        return
    
    optimizer = RAdamScheduleFree(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    train_criterion = GeneralizedDiceFocalLoss(softmax=True)
    valid_criterion = DiceScore(num_classes=CFG.mask_num, include_background=True, average=None)
    
    best_val_score = 0
    for epoch in range(CFG.num_epochs):
        print(f"\nEpoch {epoch+1}/{CFG.num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, train_criterion, device)
        val_score = validate(model, val_loader, train_criterion, valid_criterion, device, epoch, optimizer=optimizer)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_score:.4f}")
        
        if val_score > best_val_score:
            best_val_score = val_score
            state_dict = model.state_dict()
            model_path = os.path.join(base_dir, CFG.model_path)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(state_dict, model_path)
            print("Saved best model!")
    
    model_path = os.path.join(base_dir, CFG.model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test(model, test_loader, valid_criterion, device)

if __name__ == "__main__":
    main()