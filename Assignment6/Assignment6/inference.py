import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import sys


# ---------- CONFIG ----------
CKPT_PATH = "runs/colorcue/resnet18_imagenette_colorcue.pth"
IMAGE_PATH = "data/imagenette2-320-colorcue-swapped/train/n02102040/ILSVRC2012_val_00002294.JPEG"   # change this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    classes = checkpoint["classes"]

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, classes


def preprocess_image(img_path):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    img = Image.open(img_path).convert("RGB")
    img = tfm(img)
    img = img.unsqueeze(0)  # add batch dim
    return img


def main():
    image_path = IMAGE_PATH
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    model, classes = load_model(CKPT_PATH)
    x = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = torch.topk(probs, k=5)

    print("\nTop-5 Predictions:")
    for i in range(5):
        class_name = classes[top_idx[0][i].item()]
        confidence = top_prob[0][i].item() * 100
        print(f"{i+1}. {class_name} — {confidence:.2f}%")

    print("\nPredicted class:", classes[top_idx[0][0].item()])


if __name__ == "__main__":
    main()
