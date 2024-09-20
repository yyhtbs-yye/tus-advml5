from PIL import Image
import requests
from io import BytesIO
import torch

def get_image_tensor_from_url(url, transform=None):
    # Download the image
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    # Apply the same transformations
    img_transformed = transform(img) if transform else img
    
    # Convert to tensor if not already done
    if not isinstance(img_transformed, torch.Tensor):
        img_transformed = torch.tensor(np.array(img_transformed)).permute(2, 0, 1) / 255.0
    
    return img_transformed

def get_image_tensor_from_local_path(path, transform=None):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    # Apply the same transformations
    img_transformed = transform(img) if transform else img
    
    # Convert to tensor if not already done
    if not isinstance(img_transformed, torch.Tensor):
        img_transformed = torch.tensor(np.array(img_transformed)).permute(2, 0, 1) / 255.0
    
    return img_transformed

def convert_to_batch(img):
    # Ensure img is a tensor
    if not isinstance(img, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    # Unsqueeze to add a batch dimension
    if img.ndim == 3:  # RGB Scale
        img_batch = img.unsqueeze(0)  
    elif img.ndim == 4:  # Already Batched Data
        img_batch = img
    elif img.ndim == 2:  # Gray Scale
        img_batch = img.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Unsupported tensor dimensions")

    return img_batch
