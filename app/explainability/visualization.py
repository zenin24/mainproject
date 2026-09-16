"""
Explainability and visual attribution module.

Generates fine-grained spatial patch-text similarity heatmaps using
PubMedCLIP's ViT-B/32 vision encoder spatial patch tokens.
"""

import logging
import os
import numpy as np
import torch
from PIL import Image

from app.models.pubmedclip import pubmedclip_model

logger = logging.getLogger(__name__)

# Medical disclaimer attached to visual explanation metadata
EXPLAINABILITY_DISCLAIMER = (
    "The visualization represents model attention/similarity attribution "
    "and should not be interpreted as a clinically validated explanation of disease."
)


def compute_spatial_similarity_heatmap(
    image: Image.Image, text_prompt: str
) -> np.ndarray:
    """Compute 2D spatial similarity attribution heatmap.

    Extracts 49 spatial patch embeddings (7x7 grid) from PubMedCLIP ViT-B/32,
    computes cosine similarity with the target text prompt embedding, and
    upsamples to the image size.

    Args:
        image: PIL Image in RGB mode.
        text_prompt: Text prompt string (e.g., "A chest X-ray showing pneumonia").

    Returns:
        Normalized 2D float array in range [0, 1] of shape (height, width).
    """
    if not pubmedclip_model.is_loaded:
        raise RuntimeError("PubMedCLIP model is not loaded.")

    model = pubmedclip_model._model
    processor = pubmedclip_model._processor
    device = pubmedclip_model.device

    # 1. Process image and text inputs
    img_inputs = processor(images=image, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

    txt_inputs = processor(text=[text_prompt], return_tensors="pt", padding=True, truncation=True)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

    with torch.no_grad():
        # Generate normalized text embedding (1, 512)
        txt_features = model.get_text_features(**txt_inputs)
        if hasattr(txt_features, "pooler_output") and txt_features.pooler_output is not None:
            txt_features = txt_features.pooler_output
        txt_emb = txt_features / txt_features.norm(dim=-1, keepdim=True)  # (1, 512)

        # Extract patch tokens from vision encoder (1, 50, 768)
        vision_out = model.vision_model(**img_inputs)
        last_hidden_state = vision_out.last_hidden_state  # (1, 50, 768)

        # Token 0 is [CLS], tokens 1..49 are 7x7 spatial patch tokens
        patch_tokens = last_hidden_state[:, 1:, :]  # (1, 49, 768)

        # Project patch tokens to embedding space (1, 49, 512)
        patch_embeds = model.visual_projection(patch_tokens)
        patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity between each spatial patch and text prompt (1, 49)
        patch_sims = (patch_embeds @ txt_emb.T).squeeze(-1).squeeze(0)  # (49,)
        sim_grid = patch_sims.view(7, 7).cpu().numpy()  # (7, 7) grid

    # 2. Min-Max normalize 7x7 grid
    min_val, max_val = sim_grid.min(), sim_grid.max()
    if max_val > min_val:
        norm_grid = (sim_grid - min_val) / (max_val - min_val)
    else:
        norm_grid = np.zeros_like(sim_grid)

    # 3. Bilinear upsample 7x7 grid to original image size
    grid_img = Image.fromarray((norm_grid * 255).astype(np.uint8), mode="L")
    upsampled_map = grid_img.resize(image.size, resample=Image.Resampling.BILINEAR)
    heatmap_array = np.array(upsampled_map, dtype=float) / 255.0

    return heatmap_array


def generate_explanation_overlay(
    image: Image.Image,
    finding_name: str,
    text_prompt: str,
    output_path: str | None = None,
    alpha: float = 0.5,
) -> tuple[Image.Image, str]:
    """Generate visual heatmap overlay on original X-ray image and save to disk.

    Args:
        image: PIL Image in RGB mode.
        finding_name: Name of finding (e.g. 'pneumonia').
        text_prompt: Text prompt string.
        output_path: Optional file path to save the explanation PNG.
        alpha: Blend weight for heatmap overlay (0.0 to 1.0).

    Returns:
        Tuple of (Blended PIL Image, output_file_path).
    """
    heatmap_norm = compute_spatial_similarity_heatmap(image, text_prompt)

    # Colorize heatmap (Red-Yellow warm color mapping for high similarity)
    # Map [0, 1] float to RGB color map
    h_bytes = (heatmap_norm * 255).astype(np.uint8)
    
    # Try OpenCV COLORMAP_JET if available, else PIL fallback
    try:
        import cv2
        color_map = cv2.applyColorMap(h_bytes, cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(color_map)
    except Exception:
        # Pure PIL fallback: Red channel = similarity, Blue channel = inverse
        r = h_bytes
        g = (h_bytes * 0.5).astype(np.uint8)
        b = 255 - h_bytes
        rgb_arr = np.stack([r, g, b], axis=-1)
        heatmap_pil = Image.fromarray(rgb_arr, mode="RGB")

    # Blend original image with colorized heatmap
    base_rgb = image.convert("RGB")
    blended = Image.blend(base_rgb, heatmap_pil, alpha=alpha)

    if output_path is None:
        out_dir = os.path.join("data", "evaluation", "results", "explanations")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{finding_name}_explanation.png")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    blended.save(output_path, format="PNG")
    logger.info("Saved visual explanation heatmap to '%s'", output_path)

    return blended, output_path
