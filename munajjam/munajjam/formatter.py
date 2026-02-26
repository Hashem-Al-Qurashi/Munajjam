"""
Standardized JSON output formatter for Munajjam alignment results.

Provides a canonical schema for JSON output that all consumers should use
instead of hand-rolling their own dict structures.

Usage::

    from munajjam.formatter import format_results, format_to_json

    output = format_results(results, audio_file="001.wav", reciter="Badr Al-Turki")
    json_str = format_to_json(results)
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field

from munajjam.models.result import AlignmentResult
from munajjam.models.surah import SURAH_NAMES

OUTPUT_SCHEMA_VERSION = "1.0"


class FormattedSummary(BaseModel):
    """Aggregate statistics for a set of alignment results."""

    model_config = ConfigDict(frozen=True)

    average_similarity: float = Field(
        ..., description="Mean similarity score across all results",
    )
    high_confidence_count: int = Field(
        ..., description="Number of results with similarity >= 0.8",
    )
    high_confidence_percentage: float = Field(
        ..., description="Percentage of high-confidence results (0.0-100.0)",
    )
    overlap_count: int = Field(
        ..., description="Number of results with overlap detected",
    )
    total_duration: float = Field(
        ..., description="Total duration of all aligned segments in seconds",
    )


class FormattedAyahResult(BaseModel):
    """Canonical per-ayah alignment result."""

    model_config = ConfigDict(frozen=True)

    ayah_id: int = Field(
        ..., description="Global unique ayah ID (1-6236)", ge=1, le=6236,
    )
    surah_number: int = Field(
        ..., description="Surah number (1-114)", ge=1, le=114,
    )
    ayah_number: int = Field(
        ..., description="Ayah number within surah (1-based)", ge=1,
    )
    start_time: float = Field(
        ..., description="Start time in seconds", ge=0.0,
    )
    end_time: float = Field(
        ..., description="End time in seconds", ge=0.0,
    )
    duration: float = Field(
        ..., description="Duration in seconds", ge=0.0,
    )
    transcribed_text: str = Field(..., description="Text transcribed from audio")
    reference_text: str = Field(..., description="Canonical Quran text for this ayah")
    similarity_score: float = Field(
        ..., description="Similarity score (0.0-1.0)", ge=0.0, le=1.0,
    )
    is_high_confidence: bool = Field(
        ..., description="Whether similarity >= 0.8",
    )
    overlap_detected: bool = Field(
        ..., description="Whether overlap was detected",
    )


class FormattedOutput(BaseModel):
    """Top-level canonical output envelope."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(..., description="Schema version")
    surah_number: int | None = Field(
        None, description="Surah number if single-surah output",
    )
    surah_name: str | None = Field(
        None, description="Arabic surah name if single-surah output",
    )
    total_ayahs: int = Field(..., description="Number of ayahs in results")
    audio_file: str | None = Field(None, description="Path to audio file")
    reciter: str | None = Field(None, description="Reciter name")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    results: list[FormattedAyahResult] = Field(
        ..., description="Per-ayah alignment results",
    )
    summary: FormattedSummary = Field(..., description="Aggregate statistics")


def format_result(result: AlignmentResult) -> FormattedAyahResult:
    """Format a single AlignmentResult into the canonical schema."""
    rounded_start = round(result.start_time, 3)
    rounded_end = round(result.end_time, 3)

    return FormattedAyahResult(
        ayah_id=result.ayah.id,
        surah_number=result.ayah.surah_id,
        ayah_number=result.ayah.ayah_number,
        start_time=rounded_start,
        end_time=rounded_end,
        duration=max(0.0, round(rounded_end - rounded_start, 3)),
        transcribed_text=result.transcribed_text,
        reference_text=result.ayah.text,
        similarity_score=round(result.similarity_score, 4),
        is_high_confidence=result.is_high_confidence,
        overlap_detected=result.overlap_detected,
    )


def _compute_summary(
    formatted: list[FormattedAyahResult],
) -> FormattedSummary:
    """Compute aggregate summary statistics from formatted results."""
    if not formatted:
        return FormattedSummary(
            average_similarity=0.0,
            high_confidence_count=0,
            high_confidence_percentage=0.0,
            overlap_count=0,
            total_duration=0.0,
        )

    total = len(formatted)
    high_count = sum(1 for r in formatted if r.is_high_confidence)

    return FormattedSummary(
        average_similarity=round(
            sum(r.similarity_score for r in formatted) / total, 4,
        ),
        high_confidence_count=high_count,
        high_confidence_percentage=round(high_count / total * 100, 1),
        overlap_count=sum(1 for r in formatted if r.overlap_detected),
        total_duration=round(sum(r.duration for r in formatted), 3),
    )


def _detect_surah(
    formatted: list[FormattedAyahResult],
) -> tuple[int | None, str | None]:
    """Detect surah number and name if all results share the same surah."""
    if not formatted:
        return None, None

    surah_ids = {r.surah_number for r in formatted}
    if len(surah_ids) == 1:
        (surah_id,) = surah_ids
        return surah_id, SURAH_NAMES.get(surah_id)

    return None, None


def format_results(
    results: list[AlignmentResult],
    *,
    audio_file: str | None = None,
    reciter: str | None = None,
) -> FormattedOutput:
    """Format alignment results into the canonical output envelope."""
    formatted = [format_result(r) for r in results]
    surah_number, surah_name = _detect_surah(formatted)
    summary = _compute_summary(formatted)

    return FormattedOutput(
        version=OUTPUT_SCHEMA_VERSION,
        surah_number=surah_number,
        surah_name=surah_name,
        total_ayahs=len(formatted),
        audio_file=audio_file,
        reciter=reciter,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        results=formatted,
        summary=summary,
    )


def format_results_list(
    results: list[AlignmentResult],
) -> list[FormattedAyahResult]:
    """Format results as a flat list without the envelope."""
    return [format_result(r) for r in results]


def format_to_json(
    results: list[AlignmentResult],
    *,
    audio_file: str | None = None,
    reciter: str | None = None,
    indent: int = 2,
) -> str:
    """Format results directly to a JSON string."""
    output = format_results(results, audio_file=audio_file, reciter=reciter)
    return output.model_dump_json(indent=indent)
