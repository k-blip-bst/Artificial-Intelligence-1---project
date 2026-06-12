import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src import CLASSES, CLASS_TO_IDX
from src.models import build_resnet18, build_efficientnet_b0


_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def overlay_cam(original_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img = np.array(original_pil.resize((224, 224))).astype(np.uint8)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * img + (1 - alpha) * heatmap).astype(np.uint8)
    return overlay


def visualize_gradcam(
    model_path: str,
    backbone: str,
    image_paths: list[str],
    save_dir: str = "results/figures/gradcam",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone == "resnet18":
        model = build_resnet18(pretrained=False)
        target_layer = model.layer4[-1]
    else:
        model = build_efficientnet_b0(pretrained=False)
        target_layer = model.features[-1]

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    gradcam = GradCAM(model, target_layer)
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        pil_img = Image.open(img_path).convert("RGB")
        tensor = _TRANSFORM(pil_img).unsqueeze(0).to(device)
        cam, pred_idx = gradcam.generate(tensor)
        pred_label = CLASSES[pred_idx]

        overlay = overlay_cam(pil_img, cam)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.array(pil_img.resize((224, 224))))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (pred: {pred_label})")
        axes[2].axis("off")

        plt.tight_layout()
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f"{stem}_gradcam.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    gradcam.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--model_path", default="models/cnn_resnet18_best.pt")
    parser.add_argument("--images", nargs="+", required=True, help="Image file paths")
    parser.add_argument("--save_dir", default="results/figures/gradcam")
    args = parser.parse_args()
    visualize_gradcam(args.model_path, args.backbone, args.images, args.save_dir)
