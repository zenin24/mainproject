"""
Pydantic response schemas for the API.

Defines structured response models to ensure consistent JSON output.
"""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about the ML model used."""

    name: str = "PubMedCLIP"
    type: str = "zero-shot prototype"


class FindingResult(BaseModel):
    """A single finding with its similarity score, rank, and status."""

    finding: str = Field(description="Name of the medical finding")
    score: float = Field(
        description=(
            "Prototype image-text similarity score (NOT a clinically validated probability)"
        )
    )
    similarity_score: float = Field(
        description="Prototype image-text similarity score (backward compatible alias)"
    )
    rank: int = Field(description="Rank order based on similarity score (1-indexed)")
    status: str = Field(
        description="'candidate' if score >= threshold, otherwise 'unlikely'"
    )
    above_threshold: bool = Field(
        description="Whether the score is above the configured threshold"
    )
    explanation_available: bool = Field(
        default=True,
        description="Whether spatial visual explanation heatmap can be generated",
    )


class ExplainabilityInfo(BaseModel):
    """Explainability metadata and visual attribution details."""

    available: bool = True
    method: str = "CLIP-based spatial patch-text similarity attribution"
    disclaimer: str = (
        "The visualization represents model attention/similarity attribution "
        "and should not be interpreted as a clinically validated explanation of disease."
    )


class AnalysisResponse(BaseModel):
    """Complete analysis response from POST /analyze."""

    status: str = "success"
    model: ModelInfo = Field(default_factory=ModelInfo)
    findings: list[FindingResult]
    prototype_confidence: float = Field(
        description=(
            "Highest similarity score among findings. "
            "This is a prototype score, NOT a clinically validated probability."
        )
    )
    threshold_used: float = Field(
        description="The confidence threshold used for filtering"
    )
    explainability: ExplainabilityInfo = Field(default_factory=ExplainabilityInfo)
    disclaimer: str = Field(
        default="Research prototype. Not for clinical diagnosis.",
        description="Medical disclaimer",
    )


class ImageInfoResponse(BaseModel):
    """Response for basic image info (debug/testing)."""

    status: str = "success"
    filename: str
    image_format: str
    width: int
    height: int
    mode: str


class ErrorResponse(BaseModel):
    """Structured error response."""

    status: str = "error"
    detail: str
