# train_simple.py
# Simple training script to demonstrate spurious correlation learning
# The model should learn the color squares and fail on swapped validation set

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


# Configuration
TRAIN_DIR = "data/imagenette2-320-colorcue/train"
VAL_DIR = "data/imagenette2-320-colorcue/val"
OUTPUT_DIR = "runs/shortcut"
CHECKPOINT_NAME = "model.pth"

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transforms():
    """Simple transforms - no normalization, no augmentation"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def main():
    print(f"Device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs")
    print()
    
    # Setup data
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {len(train_dataset.classes)}")
    print()
    
    # Setup model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    # Training loop
    best_val_acc = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes,
            }, checkpoint_path)
    
    print()
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)}")
    print()
    print("Next steps:")
    print(f"  python evaluate_simple.py --ckpt {os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)} --data {VAL_DIR}")


if __name__ == "__main__":
    main()