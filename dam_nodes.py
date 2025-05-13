import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel
import re

class DAMObjectNameNode:
    """
    ComfyUI node that uses the NVIDIA DAM model to identify objects in masked regions
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # We'll load the model lazily when first needed
        self.model = None
        self.dam = None
    
    def _load_model(self):
        """Load the DAM model if not already loaded"""
        if self.model is None:
            print("Loading NVIDIA DAM model...")
            self.model = AutoModel.from_pretrained(
                'nvidia/DAM-3B-Self-Contained',
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            self.dam = self.model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
            print("DAM model loaded successfully")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 10, "min": 1, "max": 512, "step": 1}),
                "invert_mask": (["False", "True"], {"default": "False"}),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.999, "step": 0.001}),
                "prompt_mode": (["name_only", "full_description"], {"default": "name_only"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_object_name"
    CATEGORY = "image/analysis"
    
    def get_object_name(self, image, mask, temperature=0.1, max_tokens=10, invert_mask="False", threshold=0.01, prompt_mode="name_only"):
        """
        Process image with mask to identify object name using DAM
        """
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Convert ComfyUI tensor to PIL Image
            # ComfyUI images are in format [batch, height, width, channel]
            if len(image.shape) == 4:
                image = image[0]  # Take first image if batched
            
            # Convert PyTorch tensor to numpy array first
            image_np = image.cpu().numpy()
            # Convert from [H, W, C] to PIL Image
            # ComfyUI uses 0-1 range for pixels
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            
            # Print image information for debugging
            print(f"Image shape: {image_np.shape}")
            
            # Process the mask tensor - handle different mask formats from Load Image
            if mask is None:
                return ("No mask provided. Please connect a mask to this node.",)
                
            # Debug mask information
            print(f"Original mask type: {type(mask)}")
            print(f"Original mask shape: {mask.shape}")
            
            # Convert mask to CPU numpy array - proper handling for ComfyUI MASK format
            mask_np = mask.cpu().numpy()
            
            # Special handling for ComfyUI mask format
            # The mask from ComfyUI's Load Image can be (1, H, W) or (B, H, W, 1) or other formats
            if len(mask_np.shape) == 3:
                # This could be either (1, H, W) or (H, W, C)
                if mask_np.shape[0] == 1:
                    # This is (1, H, W), so squeeze the first dimension
                    mask_np = np.squeeze(mask_np, axis=0)
                    print(f"Squeezed mask shape: {mask_np.shape}")
                elif mask_np.shape[2] <= 4:
                    # This is likely (H, W, C) with C being 1, 3, or 4 (color channels)
                    # Convert to grayscale by taking average across channels
                    mask_np = np.mean(mask_np, axis=2)
                    print(f"Converted color mask to grayscale: {mask_np.shape}")
            elif len(mask_np.shape) == 4:
                # This is likely (B, H, W, C)
                if mask_np.shape[0] == 1:
                    # Take the first batch
                    mask_np = mask_np[0]
                    # If it still has a channel dimension, take the average
                    if len(mask_np.shape) == 3 and mask_np.shape[2] <= 4:
                        mask_np = np.mean(mask_np, axis=2)
            
            # Print processed mask shape
            print(f"Processed mask shape: {mask_np.shape}")
            print(f"Mask min: {np.min(mask_np)}, max: {np.max(mask_np)}, mean: {np.mean(mask_np)}")
            
            # Now check that mask_np is 2D
            if len(mask_np.shape) != 2:
                print(f"WARNING: Mask has unexpected dimensions {mask_np.shape}. Attempting to convert to 2D.")
                # Try to reshape or reduce dimensions
                if len(mask_np.shape) == 3 and mask_np.shape[2] == 1:
                    # If it's (H, W, 1), squeeze the last dimension
                    mask_np = np.squeeze(mask_np, axis=2)
                elif len(mask_np.shape) == 3:
                    # If it's (H, W, C), take the first channel
                    mask_np = mask_np[:, :, 0]
                elif len(mask_np.shape) == 1:
                    # If it's (N,), try to reshape to a square
                    size = int(np.sqrt(mask_np.shape[0]))
                    mask_np = mask_np[:size*size].reshape(size, size)
            
            # After all processing, ensure mask is 2D
            if len(mask_np.shape) != 2:
                return (f"Error: Unable to convert mask to 2D. Current shape: {mask_np.shape}",)
            
            # Handle mask inversion if needed
            if invert_mask == "True":
                print("Inverting mask")
                mask_np = 1.0 - mask_np
            
            # Creating a binary mask with the specified threshold
            # This is critical for faint masks from Load Image
            print(f"Using threshold: {threshold}")
            mask_binary = (mask_np > threshold).astype(np.float32)
            
            # Log mask information after thresholding
            white_pixels = np.sum(mask_binary > 0)
            total_pixels = mask_binary.size
            white_percentage = (white_pixels / total_pixels) * 100
            print(f"Binary mask has {white_pixels} white pixels out of {total_pixels} total pixels ({white_percentage:.2f}%)")
            
            # Check if mask is still empty after processing
            if white_pixels == 0:
                return (f"Mask is empty after processing with threshold {threshold}. Please try a lower threshold or check that your mask contains visible white areas.",)
            
            # Save debug mask image to help diagnose issues
            try:
                debug_path = os.path.join(os.path.dirname(__file__), "dam_debug_mask.png")
                debug_mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8), mode='L')
                debug_mask_img.save(debug_path)
                print(f"Saved debug mask to {debug_path}")
            except Exception as save_err:
                print(f"Warning: Could not save debug mask: {str(save_err)}")
            
            # Convert binary mask to PIL Image for DAM
            pil_mask = Image.fromarray((mask_binary * 255).astype(np.uint8), mode='L')
            
            # Make sure mask is the same size as the image
            if pil_mask.size != pil_image.size:
                print(f"Resizing mask from {pil_mask.size} to match image size {pil_image.size}")
                pil_mask = pil_mask.resize(pil_image.size, Image.NEAREST)
            
            # Define prompt based on prompt_mode
            if prompt_mode == "name_only":
                prompt = '<image>\nWhat is the name of this object? Answer with a single word.'
                # Use fewer tokens for name-only mode
                actual_max_tokens = min(max_tokens, 10)
            else:
                prompt = '<image>\nDescribe the masked region in detail.'
                # Use the full token count for descriptions
                actual_max_tokens = max_tokens
            
            print(f"Using prompt: '{prompt}' with max_tokens: {actual_max_tokens}")
            
            # Get description from DAM model
            result = ""
            for token in self.dam.get_description(
                pil_image,
                pil_mask,
                prompt,
                streaming=False,  # Set to False for ComfyUI integration
                temperature=temperature,
                top_p=0.5,
                num_beams=1,
                max_new_tokens=actual_max_tokens
            ):
                result += token
            
            # Clean up the result for name_only mode
            if prompt_mode == "name_only":
                # Remove articles (a, an, the)
                result = re.sub(r'^(a|an|the)\s+', '', result.lower().strip(), flags=re.IGNORECASE)
                # Remove punctuation
                result = re.sub(r'[^\w\s]', '', result).strip()
            
            print(f"DAM result: {result}")
            
            return (result,)
            
        except Exception as e:
            print(f"Error in DAM model processing: {str(e)}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"Error: {str(e)}"
            # Add additional diagnostic info if available
            if 'mask_np' in locals() and 'white_pixels' in locals():
                error_msg += f" Mask has {white_pixels} white pixels out of {total_pixels} ({white_percentage:.2f}%)."
            
            return (error_msg,)


class DAMVisualizeNode:
    """
    ComfyUI node that adds contours to visualize the mask region
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "contour_thickness": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "contour_color": (["white", "red", "green", "blue", "yellow"], {"default": "white"}),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.999, "step": 0.001}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_contour"
    CATEGORY = "image/visualization"
    
    def add_contour(self, image, mask, contour_thickness=6, contour_color="white", threshold=0.01):
        """
        Add contour to the image based on the mask
        """
        try:
            # Import opencv here to avoid dependency issues if not installed
            import cv2
            
            # Convert ComfyUI tensor to numpy array
            # Convert ComfyUI tensor to numpy array
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 4:
                    image = image[0]
                image_np = image.cpu().numpy()
            else:
                if len(image.shape) == 4:
                    image = image[0]
                image_np = image  # already numpy

            # Same for mask
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask

            # Process the mask to ensure it's 2D
            if len(mask_np.shape) == 3:
                # This could be either (1, H, W) or (H, W, C)
                if mask_np.shape[0] == 1:
                    # This is (1, H, W), so squeeze the first dimension
                    mask_np = np.squeeze(mask_np, axis=0)
                elif mask_np.shape[2] <= 4:
                    # This is likely (H, W, C) with C being 1, 3, or 4
                    mask_np = np.mean(mask_np, axis=2)
            elif len(mask_np.shape) == 4:
                # This is likely (B, H, W, C)
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                    if len(mask_np.shape) == 3 and mask_np.shape[2] <= 4:
                        mask_np = np.mean(mask_np, axis=2)
            
            # After all processing, ensure mask is 2D
            if len(mask_np.shape) != 2:
                print(f"WARNING: Mask for visualization has unexpected dimensions {mask_np.shape}. Using original image.")
                return (image[None, :, :, :] if len(image.shape) == 3 else image,)
            
            # Convert to binary mask for findContours
            mask_for_contour = (mask_np > threshold).astype(np.uint8) * 255
            
            # Define color mapping
            color_map = {
                "white": (1.0, 1.0, 1.0),
                "red": (1.0, 0.0, 0.0),
                "green": (0.0, 1.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
                "yellow": (1.0, 1.0, 0.0)
            }
            contour_color_rgb = color_map.get(contour_color, (1.0, 1.0, 1.0))
            
            # Create a copy of the image to draw on
            img_with_contour = image_np.copy()
            
            # Find contours
            contours, _ = cv2.findContours(mask_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Log contour information
            print(f"Found {len(contours)} contours")
            
            # Draw contours
            cv2.drawContours(img_with_contour, contours, -1, contour_color_rgb, thickness=contour_thickness)
            
            # Return as ComfyUI tensor format [B, H, W, C]
            return (torch.from_numpy(img_with_contour).unsqueeze(0).float(),)
            
        except Exception as e:
            print(f"Error in contour visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original image if there's an error
            return (image[None, :, :, :] if len(image.shape) == 3 else image,)


# This part is needed for ComfyUI to discover and register the nodes
NODE_CLASS_MAPPINGS = {
    "DAMObjectNameNode": DAMObjectNameNode,
    "DAMVisualizeNode": DAMVisualizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAMObjectNameNode": "DAM Object Name Extractor",
    "DAMVisualizeNode": "DAM Mask Visualizer"
}
