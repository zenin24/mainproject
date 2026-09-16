"""
API route definitions.

This module contains all FastAPI route handlers for the Chest X-ray Finding Analyzer.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.inference.analyzer import analyze_image
from app.models.pubmedclip import pubmedclip_model
from app.preprocessing.image import (
    get_image_info,
    load_image,
    validate_image_file,
)
from app.schemas.response import (
    AnalysisResponse,
    ErrorResponse,
    ExplainabilityInfo,
    FindingResult,
    ImageInfoResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint for basic status check."""
    return {
        "message": "Chest X-ray Finding Analyzer API",
        "status": "running",
        "model_loaded": pubmedclip_model.is_loaded,
    }


@router.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Detailed health check endpoint reporting model status and compute device."""
    return {
        "status": "healthy",
        "model": pubmedclip_model.get_model_info(),
    }


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file format or corrupted image"},
        500: {"model": ErrorResponse, "description": "Inference failure"},
    },
    tags=["Analysis"],
)
async def analyze_xray(file: UploadFile = File(...)) -> AnalysisResponse:
    """Analyze a chest X-ray image for medical findings using PubMedCLIP.

    - **file**: Chest X-ray image file (JPEG, PNG, BMP, TIFF)

    Returns ranked medical findings with prototype similarity scores, rank, and candidate status.
    """
    logger.info("Received image analysis request: filename=%s, content_type=%s", file.filename, file.content_type)

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload request.",
        )

    # Step 1: Validate file extension and MIME type
    try:
        validate_image_file(file.filename, file.content_type)
    except ValueError as e:
        logger.warning("Image validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Step 2: Read raw bytes and load image into PIL RGB format
    try:
        contents = await file.read()
        image = load_image(contents)
    except ValueError as e:
        logger.warning("Failed to parse image file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupt image file: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error reading image upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded image.",
        )

    # Step 3: Run inference pipeline
    try:
        analysis_result = analyze_image(image)
    except RuntimeError as e:
        logger.error("Model inference error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Unexpected error during analysis: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during X-ray analysis: {e}",
        )

    # Step 4: Construct structured response with rank, score, and status
    findings = [
        FindingResult(
            finding=f["finding"],
            score=f["score"],
            similarity_score=f["similarity_score"],
            rank=f["rank"],
            status=f["status"],
            above_threshold=f["above_threshold"],
            explanation_available=f.get("explanation_available", True),
        )
        for f in analysis_result["findings"]
    ]

    return AnalysisResponse(
        status="success",
        model=ModelInfo(),
        findings=findings,
        prototype_confidence=analysis_result["prototype_confidence"],
        threshold_used=analysis_result["threshold_used"],
        explainability=ExplainabilityInfo(),
        disclaimer="Research prototype. Not for clinical diagnosis.",
    )


@router.post(
    "/explain",
    responses={
        200: {"content": {"image/png": {}}, "description": "Visual heatmap overlay PNG image"},
        400: {"model": ErrorResponse, "description": "Invalid image file or parameters"},
        500: {"model": ErrorResponse, "description": "Heatmap generation failure"},
    },
    tags=["Explainability"],
)
async def generate_explainability_map(
    file: UploadFile = File(...),
    finding: str = "pneumonia",
):
    """Generate a visual spatial patch-text similarity heatmap overlay PNG for a specific finding."""
    from io import BytesIO
    from fastapi.responses import Response
    from app.explainability.visualization import generate_explanation_overlay
    from app.inference.analyzer import FINDING_PROMPTS

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        validate_image_file(file.filename, file.content_type)
        contents = await file.read()
        image = load_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Match prompt text for given finding
    prompt_map = {f_name: p_text for f_name, p_text in FINDING_PROMPTS}
    text_prompt = prompt_map.get(
        finding, f"A chest X-ray showing {finding.replace('_', ' ')}"
    )

    try:
        blended_img, _ = generate_explanation_overlay(image, finding, text_prompt)
        buf = BytesIO()
        blended_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Failed to generate explainability overlay: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate explainability map: {e}"
        )


@router.post(
    "/image-info",
    response_model=ImageInfoResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Utility"],
)
async def inspect_image(file: UploadFile = File(...)) -> ImageInfoResponse:
    """Inspect basic metadata of an uploaded image file without running inference."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        validate_image_file(file.filename, file.content_type)
        contents = await file.read()
        image = load_image(contents)
        info = get_image_info(image)
        return ImageInfoResponse(
            status="success",
            filename=file.filename,
            image_format=info["format"],
            width=info["width"],
            height=info["height"],
            mode=info["mode"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

