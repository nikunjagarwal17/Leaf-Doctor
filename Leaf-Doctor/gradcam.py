"""
GradCAM (Gradient-weighted Class Activation Mapping) implementation for CNN visualization.
Generates heatmaps showing which regions of the leaf image contributed most to the prediction.
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Tuple

# Set non-GUI backend before importing matplotlib to avoid threading issues in Flask
import matplotlib
matplotlib.use('Agg')

# Suppress expected warnings
warnings.filterwarnings('ignore', message='Full backward hook is firing')


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretability.
    Highlights regions of the image that most influenced the model's prediction.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "conv_layers"):
        """
        Initialize GradCAM.
        
        Args:
            model: The CNN model to analyze
            target_layer_name: Name of the layer to generate activations from
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        # Get the target layer
        target_layer = getattr(self.model, self.target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
        
        # Forward hook to capture feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        device: torch.device = torch.device("cpu")
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for a given image and target class.
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class index for which to generate CAM
            device: Device to run on (cpu or cuda)
        
        Returns:
            Heatmap as numpy array (224, 224)
        """
        self.model.eval()
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create target vector
        target = torch.zeros(output.shape)
        target[0][target_class] = 1.0
        target = target.to(device)
        
        # Backward pass
        output.backward(gradient=target)
        
        # Get feature maps and gradients
        feature_maps = self.feature_maps
        gradients = self.gradients
        
        if feature_maps is None or gradients is None:
            raise RuntimeError("Could not capture feature maps or gradients")
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Apply weights to feature maps
        weighted_maps = feature_maps * weights
        
        # Sum across channel dimension
        heatmap = torch.sum(weighted_maps, dim=1).squeeze(0)
        
        # Apply ReLU to keep only positive contributions
        heatmap = F.relu(heatmap)
        
        # Normalize to [0, 1]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        
        if heatmap_max - heatmap_min != 0:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = torch.zeros_like(heatmap)
        
        # Resize to original image size (224x224)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().cpu().numpy()
        
        return heatmap


def overlay_heatmap_on_image(
    image_path: Path,
    heatmap: np.ndarray,
    output_path: Path = None,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> Tuple[Image.Image, Path]:
    """
    Overlay GradCAM heatmap on the original image.
    
    Args:
        image_path: Path to the original image
        heatmap: GradCAM heatmap (224, 224)
        output_path: Path to save the overlay image
        alpha: Transparency of heatmap overlay (0-1)
        colormap: Colormap to use for heatmap visualization
    
    Returns:
        Tuple of (PIL Image, output path)
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    
    # Convert heatmap to PIL Image with colormap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Normalize heatmap to [0, 1] if not already
    if heatmap.max() > 1:
        heatmap = heatmap / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Get RGB, drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_colored)
    
    # Blend images
    overlay = Image.blend(image, heatmap_image, alpha)
    
    # Save if output path provided
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_gradcam.png"
    
    overlay.save(output_path)
    
    return overlay, output_path


def create_comparison_image(
    image_path: Path,
    heatmap: np.ndarray,
    output_path: Path = None
) -> Tuple[Image.Image, Path]:
    """
    Create a side-by-side comparison of original image and heatmap.
    
    Args:
        image_path: Path to the original image
        heatmap: GradCAM heatmap
        output_path: Path to save the comparison image
    
    Returns:
        Tuple of (PIL Image, output path)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Load original image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    
    # Normalize heatmap
    if heatmap.max() > 1:
        heatmap = heatmap / 255.0
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap
    cmap = cm.get_cmap('jet')
    axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_comparison.png"
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Load saved figure as PIL Image
    result_image = Image.open(output_path)
    
    return result_image, output_path
