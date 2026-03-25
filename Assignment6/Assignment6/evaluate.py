# evaluate_simple.py
# Simple evaluation script - uses same transforms as training

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


def get_transforms():
    """Must match training transforms exactly"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    transform = get_transforms()
    dataset = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    checkpoint = torch.load(args.ckpt, map_location=device)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    accuracy = evaluate(model, loader, device)
    
    print(f"Dataset: {args.data}")
    print(f"Samples: {len(dataset)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()