import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _resize_preserve(image_pil, size):
    return image_pil.resize((size, size), Image.BILINEAR)


def apply_filter(img_pil: Image.Image, mode: str = "bilateral") -> Image.Image:
    arr = np.array(img_pil.convert("RGB"))
    if mode == "bilateral":
        arr = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    elif mode == "clahe":
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif mode == "histeq":
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        arr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    elif mode == "gaussian":
        arr = cv2.GaussianBlur(arr, (5, 5), 0)
    return Image.fromarray(arr)


class OvarianDataset(Dataset):
    def __init__(self, root: str | Path, split: str, cfg, transforms_=None, indices=None):
        self.root = Path(root)
        self.cfg = cfg
        self.split = split
        self.imgs = []
        self.labels = []
        self.cache = {}
        is_train = split == "train"
        for label, cls in enumerate(sorted(p.name for p in self.root.glob("*") if p.is_dir())):
            files = sorted((self.root / cls).glob("*.jpg")) + sorted((self.root / cls).glob("*.jpeg")) + sorted((self.root / cls).glob("*.png"))
            for f in files:
                self.imgs.append(f)
                self.labels.append(label)
        if indices is not None:
            self.imgs = [self.imgs[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        if transforms_ is not None:
            self.transform = transforms_
        else:
            self.transform = self._build_transform(is_train)

    def _build_transform(self, is_train):
        if is_train and self.cfg.augment:
            aug = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=self.cfg.rotation),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ColorJitter(
                        brightness=self.cfg.brightness,
                        contrast=self.cfg.contrast,
                        saturation=self.cfg.saturation,
                        hue=self.cfg.hue,
                    ),
                    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                    transforms.ToTensor(),
                    transforms.Lambda(self._sharpen),
                ]
            )
            return transforms.Compose([aug, transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _sharpen(self, x):
        if random.random() < 0.3:
            x = x * 255.0
            k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=x.dtype)
            kernel = k[None, None].expand(3, 1, 3, 3)
            x = torch.nn.functional.conv2d(x[None], kernel, padding=1, groups=3)
            return torch.clamp(x[0], 0, 255) / 255.0
        return x

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if idx in self.cache:
            img = self.cache[idx]
        else:
            img = Image.open(self.imgs[idx]).convert("RGB")
            img = apply_filter(img, self.cfg.filter_mode)
            img = _resize_preserve(img, self.cfg.img_size)
            if len(self.cache) < 500:
                self.cache[idx] = img
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def strat_split(cfg, num_samples, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_ratio + cfg.test_ratio, random_state=cfg.seed)
    train_idx, tmp_idx = next(sss.split(range(num_samples), labels))
    tmp_labels = [labels[i] for i in tmp_idx]
    ratio = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio + 1e-12)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=cfg.seed)
    val_idx, test_idx = next(sss2.split(tmp_idx, tmp_labels))
    return train_idx, [tmp_idx[i] for i in val_idx], [tmp_idx[i] for i in test_idx]


def build_datasets(cfg, root="Dataset"):
    root = Path(root)
    all_imgs = []
    all_labels = []
    for label, cls in enumerate(sorted(p.name for p in root.glob("*") if p.is_dir())):
        files = sorted((root / cls).glob("*.jpg")) + sorted((root / cls).glob("*.jpeg")) + sorted((root / cls).glob("*.png"))
        for f in files:
            all_imgs.append(f)
            all_labels.append(label)
    train_idx, val_idx, test_idx = strat_split(cfg, len(all_imgs), all_labels)
    train_ds = OvarianDataset(root, "train", cfg, indices=train_idx)
    val_ds = OvarianDataset(root, "val", cfg, indices=val_idx)
    test_ds = OvarianDataset(root, "test", cfg, indices=test_idx)
    return train_ds, val_ds, test_ds


def make_dataloaders(cfg, train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=cfg.fe_batch, shuffle=True, num_workers=cfg.fe_num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.fe_batch, shuffle=False, num_workers=cfg.fe_num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.fe_batch, shuffle=False, num_workers=cfg.fe_num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader