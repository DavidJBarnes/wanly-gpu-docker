"""
Compatibility shim for missing comfy_api_nodes.util functions.
"""
from io import BytesIO
import torch
import numpy as np
from PIL import Image


def bytesio_to_image_tensor(buffer: BytesIO) -> torch.Tensor:
    """Convert BytesIO image to tensor in ComfyUI format [B, H, W, C]."""
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    img_array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dim
    return tensor


def tensor_to_bytesio(tensor: torch.Tensor, mime_type: str = 'image/png') -> BytesIO:
    """Convert tensor in ComfyUI format [B, H, W, C] to BytesIO."""
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Convert to numpy and scale to 0-255
    img_array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Create PIL image
    image = Image.fromarray(img_array)
    
    # Save to BytesIO
    buffer = BytesIO()
    format_map = {
        'image/png': 'PNG',
        'image/jpeg': 'JPEG', 
        'image/webp': 'WEBP'
    }
    fmt = format_map.get(mime_type, 'PNG')
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer
