import logging
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms  # type: ignore

# Flexible Grad-CAM import for stable operation in different environments
try:
    from pytorch_grad_cam import GradCAM  # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
except ImportError:
    GradCAM = None
    show_cam_on_image = None

logger = logging.getLogger(__name__)


def get_inference_transform() -> transforms.Compose:
    """
    Standard preprocessing pipeline for CIFAR-10 inference.
    Converts PIL Image to normalized PyTorch tensor.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(
    model: torch.nn.Module, img: Image.Image
) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
    """
    Performs model inference on a single image.

    Args:
        model: Trained PyTorch model.
        img: Input PIL Image.

    Returns:
        tuple: (predicted_class_index, confidence_score, input_tensor, all_probabilities)
    """
    logger.info("Executing inference for a single sample.")

    transform = get_inference_transform()
    img_t = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_t)
        probs = F.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)

    logger.info(
        f"Inference result: class {pred_idx.item()} "
        f"with {confidence.item():.2%} confidence."
    )

    return int(pred_idx.item()), float(confidence.item()), img_t, probs


def run_gradcam(
    model: torch.nn.Module, img_tensor: torch.Tensor, original_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Generates Grad-CAM visual explanation.

    Args:
        model: Trained model.
        img_tensor: Preprocessed input tensor.
        original_size: Target size (width, height) to match the original image aspect ratio.

    Returns:
        Optional[np.ndarray]: Colored heatmap overlaid on image or None if failed.
    """
    if GradCAM is None or show_cam_on_image is None:
        logger.warning("Grad-CAM library not found. Skipping visualization.")
        return None

    # Identify the target layer for visualization (usually the last convolutional layer)
    target_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ][-1:]

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=img_tensor)[0, :]

        # Prepare background image from tensor (convert back to RGB 0-1)
        rgb_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        # Denormalize using ImageNet stats
        rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array(
            [0.485, 0.456, 0.406]
        )
        rgb_img = np.clip(rgb_img, 0, 1)

        # Merge heatmap and original image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Resize to match the UI element size or original upload size
        return cv2.resize(cam_image, original_size, interpolation=cv2.INTER_CUBIC)

    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM: {e}")
        return None
