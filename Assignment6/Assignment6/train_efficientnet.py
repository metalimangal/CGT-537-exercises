# train_efficientnet.py
import os, random
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0


@dataclass
class Config:
    train_root = "data/imagenette2-320-colorcue/train"
    val_root   = "data/imagenette2-320-colorcue/val"
    out_dir    = "runs/effnet_colorcue"
    ckpt_name  = "efficientnet_b0_imagenette_colorcue.pth"

    image_size: int = 224
    resize_shorter: int = 256

    epochs: int = 20
    batch_size: int = 32  # start conservative on Windows
    num_workers: int = 0
    pin_memory: bool = False

    lr: float = 0.05       # EfficientNet often likes a bit smaller LR than ResNet18
    momentum: float = 0.9
    weight_decay: float = 1e-4
    step_size: int = 10
    gamma: float = 0.1

    seed: int = 42
    use_cuda_if_available: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(cfg: Config) -> torch.device:
    return torch.device("cuda") if (cfg.use_cuda_if_available and torch.cuda.is_available()) else torch.device("cpu")


def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tfm = transforms.Compose([
        transforms.Resize(cfg.resize_shorter),
        transforms.RandomCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(cfg.resize_shorter),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfm, val_tfm


def build_loaders(cfg: Config):
    train_tfm, val_tfm = build_transforms(cfg)
    train_ds = datasets.ImageFolder(cfg.train_root, transform=train_tfm)
    val_ds   = datasets.ImageFolder(cfg.val_root, transform=val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, persistent_workers=False
    )
    return train_loader, val_loader, train_ds, val_ds


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = efficientnet_b0(weights=None)  # from scratch
    # Replace classifier head
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    model.to(device)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / total, correct / total


def save_checkpoint(path: str, model: nn.Module, classes: list, cfg: Config, extra: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "config": cfg.__dict__,
        **extra,
    }, path)


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device(cfg)
    print("Device:", device)

    train_loader, val_loader, train_ds, val_ds = build_loaders(cfg)
    num_classes = len(train_ds.classes)
    print("Train:", len(train_ds), "Val:", len(val_ds), "Classes:", num_classes)

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    best_val_acc = -1.0
    best_path = os.path.join(cfg.out_dir, cfg.ckpt_name)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train acc {tr_acc:.4f} | val acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(best_path, model, train_ds.classes, cfg, {"best_val_acc": best_val_acc, "best_epoch": epoch})

    print("Best val acc:", best_val_acc)
    print("Saved:", best_path)


if __name__ == "__main__":
    main()
