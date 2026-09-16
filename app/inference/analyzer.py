"""
Finding analyzer — orchestrates the full inference pipeline.

Takes a preprocessed image, runs it through PubMedCLIP against
controlled medical finding prompts, and returns ranked results.
"""

import logging

from PIL import Image

from app.config import settings
from app.models.pubmedclip import pubmedclip_model

logger = logging.getLogger(__name__)

# Controlled medical finding prompts.
# Each tuple: (finding_name, prompt_text)
# The prompt format is designed for CLIP-style zero-shot classification.
FINDING_PROMPTS: list[tuple[str, str]] = [
    ("normal", "A chest X-ray showing a normal healthy lung"),
    ("pneumonia", "A chest X-ray showing pneumonia"),
    ("pleural_effusion", "A chest X-ray showing pleural effusion"),
    ("pneumothorax", "A chest X-ray showing pneumothorax"),
    ("cardiomegaly", "A chest X-ray showing cardiomegaly"),
    ("atelectasis", "A chest X-ray showing atelectasis"),
    ("pulmonary_edema", "A chest X-ray showing pulmonary edema"),
    ("consolidation", "A chest X-ray showing consolidation"),
]


def get_finding_names() -> list[str]:
    """Return the list of finding names."""
    return [name for name, _ in FINDING_PROMPTS]


def get_prompt_texts() -> list[str]:
    """Return the list of prompt texts for embedding."""
    return [prompt for _, prompt in FINDING_PROMPTS]


def analyze_image(image: Image.Image) -> dict:
    """Run the full analysis pipeline on a chest X-ray image.

    Pipeline:
        1. Generate image embedding via PubMedCLIP
        2. Generate text embeddings for all finding prompts
        3. Compute cosine similarity
        4. Rank findings by similarity score
        5. Apply confidence threshold and top-k filtering
        6. Format structured results with rank, score, and candidate status

    Args:
        image: Preprocessed PIL Image in RGB mode.

    Returns:
        Dictionary with ranked findings and metadata.

    Raises:
        RuntimeError: If the model is not loaded.
    """
    if not pubmedclip_model.is_loaded:
        raise RuntimeError("PubMedCLIP model is not loaded.")

    finding_names = get_finding_names()
    prompt_texts = get_prompt_texts()

    logger.info("Generating image embedding...")
    image_embedding = pubmedclip_model.get_image_embedding(image)

    logger.info("Generating text embeddings for %d prompts...", len(prompt_texts))
    text_embeddings = pubmedclip_model.get_text_embeddings(prompt_texts)

    logger.info("Computing similarity scores...")
    similarities = pubmedclip_model.compute_similarity(image_embedding, text_embeddings)

    # Convert to Python floats and pair with finding names
    scores = similarities.cpu().tolist()
    results = list(zip(finding_names, scores))

    # Sort by similarity score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Apply top-k filter
    top_k = settings.TOP_K
    results = results[:top_k]

    # Build structured findings list with rank and candidate status
    findings = []
    for rank_idx, (finding_name, score) in enumerate(results, start=1):
        rounded_score = round(score, 4)
        above = rounded_score >= settings.CONFIDENCE_THRESHOLD
        findings.append({
            "finding": finding_name,
            "score": rounded_score,
            "similarity_score": rounded_score,
            "rank": rank_idx,
            "status": "candidate" if above else "unlikely",
            "above_threshold": above,
            "explanation_available": True,
        })

    # Prototype confidence is the highest similarity score
    prototype_confidence = round(results[0][1], 4) if results else 0.0

    return {
        "findings": findings,
        "prototype_confidence": prototype_confidence,
        "threshold_used": settings.CONFIDENCE_THRESHOLD,
    }
