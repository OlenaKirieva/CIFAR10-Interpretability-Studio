import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

cv2: Any = None
lime_image: Any = None
mark_boundaries: Any = None
GradCAM: Any = None
show_cam_on_image: Any = None

try:
    import cv2 as _cv2  # type: ignore

    cv2 = _cv2
except ImportError:
    pass

try:
    from lime import lime_image as _li  # type: ignore
    from skimage.segmentation import mark_boundaries as _mb  # type: ignore

    lime_image = _li
    mark_boundaries = _mb
except ImportError:
    pass

try:
    from pytorch_grad_cam import GradCAM as _gc  # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image as _sci  # type: ignore

    GradCAM = _gc
    show_cam_on_image = _sci
except ImportError:
    pass

logger = logging.getLogger(__name__)


def run_gradcam(
    model: torch.nn.Module, img_tensor: torch.Tensor, original_size: tuple
) -> Optional[np.ndarray]:
    """Grad-CAM with Block3 targeting for better spatial detail."""
    if GradCAM is None or cv2 is None:
        return None

    # Target intermediate layer for more diverse patterns
    if hasattr(model, "block3"):
        target_layers = [model.block3[-3]]
    elif hasattr(model, "conv2"):
        target_layers = [model.conv2]
    else:
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        target_layers = [conv_layers[-2]] if len(conv_layers) > 1 else [conv_layers[-1]]

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=img_tensor)[0, :]

        rgb_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array(
            [0.485, 0.456, 0.406]
        )
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return cv2.resize(cam_image, original_size, interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        return None


def run_lime(model: torch.nn.Module, img_pil: Image.Image) -> Optional[np.ndarray]:
    """Optimized LIME with HD segmentation and low-res proxy prediction."""
    if lime_image is None or mark_boundaries is None or cv2 is None:
        return None

    img_hd = img_pil.resize((224, 224))
    img_array = np.array(img_hd)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        from src.inference import get_inference_transform

        model.eval()
        batch = torch.stack(
            [get_inference_transform()(Image.fromarray(i)) for i in images]
        )
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1)
        return probs.cpu().numpy()

    try:
        explanation = explainer.explain_instance(
            img_array, predict_fn, top_labels=1, hide_color=0, num_samples=150
        )
        _, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        res_img = mark_boundaries(img_array / 255.0, mask, color=(0, 1, 0), mode="subpixel")
        return (res_img * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"LIME failed: {e}")
        return None
