import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from train_resnet50_scratch import build_scratch_resnet50
from pretrained_resnet50_experiment_architecture import build_modified_pretrained_resnet50


CLASS_NAMES = ["Egg (Food)", "Chicken", "Balloon"]
IMG_SIZE = 224

def build_pretrained_resnet50(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    target_layer = model.layer4[-1].conv3
    return model, target_layer

def load_model(model_type: str, checkpoint_path: Path, device):
    if model_type == "scratch":
        model = build_scratch_resnet50(len(CLASS_NAMES))
        target_layer = model.layer4[-1].conv3
    elif model_type == "pretrained":
        model, target_layer = build_pretrained_resnet50(len(CLASS_NAMES))
    elif model_type == "modified_pretrained":
        model, target_layer = build_modified_pretrained_resnet50(len(CLASS_NAMES))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, target_layer


def preprocess_image(image_path: Path, device):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Für Visualisierung (Grad-CAM Overlay)
    rgb_img = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    return rgb_img, input_tensor


def generate_gradcam(model, target_layer, input_tensor):
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    predicted_class = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, predicted_class].item())

    model.zero_grad()
    score = output[:, predicted_class]
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    activation = activations[0][0]
    gradient = gradients[0][0]

    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = (weights * activation).sum(dim=0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.cpu().numpy(), predicted_class, confidence


def overlay_heatmap(rgb_img, cam):
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((rgb_img.shape[1], rgb_img.shape[0]))
    cam_resized = np.array(cam_resized).astype(np.float32) / 255.0

    heatmap = plt.get_cmap("jet")(cam_resized)[..., :3]
    overlay = 0.4 * heatmap + 0.6 * rgb_img
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_visualization(image_path: Path, rgb_img, overlay, predicted_class: int, confidence: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_gradcam.png"

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM: {CLASS_NAMES[predicted_class]} ({confidence*100:.1f}%)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM activation maps for local images.")
    parser.add_argument("--model-type", choices=["scratch", "pretrained", "modified_pretrained"], required=True)
    parser.add_argument("--checkpoint", default= None, help="Path to the trained .pth checkpoint file")
    parser.add_argument("--images-dir", default="images", help="Directory with your test images")
    parser.add_argument("--output-dir", default="results/gradcam", help="Directory to save Grad-CAM outputs")
    args = parser.parse_args()

    if args.model_type != "pretrained" and args.checkpoint is None:
       raise ValueError("Checkpoint required for scratch and modified_pretrained models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else None
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if checkpoint_path is not None and not checkpoint_path.exists():
       raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    model, target_layer = load_model(args.model_type, checkpoint_path, device)

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in {images_dir}")

    for image_path in image_paths:
        rgb_img, input_tensor = preprocess_image(image_path, device)
        cam, predicted_class, confidence = generate_gradcam(model, target_layer, input_tensor)
        overlay = overlay_heatmap(rgb_img, cam)
        out_path = save_visualization(image_path, rgb_img, overlay, predicted_class, confidence, output_dir)
        print(f"Saved Grad-CAM for {image_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
