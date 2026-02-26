"""
Tests for the standardized JSON output formatter.

Verifies the canonical schema, formatting functions, rounding precision,
metadata handling, serialization round-trips, and edge cases.
"""

import json
from datetime import datetime

import pytest

from munajjam.formatter import (
    FormattedAyahResult,
    FormattedOutput,
    format_result,
    format_results,
    format_results_list,
    format_to_json,
)
from munajjam.models import AlignmentResult, Ayah

# --------------- Fixtures ---------------


@pytest.fixture()
def ayah_1() -> Ayah:
    return Ayah(
        id=1,
        surah_id=1,
        ayah_number=1,
        text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    )


@pytest.fixture()
def ayah_2() -> Ayah:
    return Ayah(
        id=2,
        surah_id=1,
        ayah_number=2,
        text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    )


@pytest.fixture()
def ayah_other_surah() -> Ayah:
    return Ayah(
        id=6231,
        surah_id=114,
        ayah_number=1,
        text="قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
    )


@pytest.fixture()
def result_1(ayah_1: Ayah) -> AlignmentResult:
    return AlignmentResult(
        ayah=ayah_1,
        start_time=5.1234567,
        end_time=8.9876543,
        transcribed_text="بسم الله الرحمن الرحيم",
        similarity_score=0.95123,
    )


@pytest.fixture()
def result_2(ayah_2: Ayah) -> AlignmentResult:
    return AlignmentResult(
        ayah=ayah_2,
        start_time=9.0,
        end_time=13.5,
        transcribed_text="الحمد لله رب العالمين",
        similarity_score=0.72,
        overlap_detected=True,
    )


@pytest.fixture()
def result_other_surah(ayah_other_surah: Ayah) -> AlignmentResult:
    return AlignmentResult(
        ayah=ayah_other_surah,
        start_time=0.0,
        end_time=3.0,
        transcribed_text="قل اعوذ برب الناس",
        similarity_score=0.88,
    )


# --------------- format_result() ---------------


class TestFormatResult:
    """Tests for formatting a single AlignmentResult."""

    def test_returns_formatted_ayah_result(
        self, result_1: AlignmentResult,
    ) -> None:
        formatted = format_result(result_1)
        assert isinstance(formatted, FormattedAyahResult)

    def test_ayah_fields_mapped(
        self, result_1: AlignmentResult, ayah_1: Ayah,
    ) -> None:
        formatted = format_result(result_1)
        assert formatted.ayah_id == ayah_1.id
        assert formatted.surah_number == ayah_1.surah_id
        assert formatted.ayah_number == ayah_1.ayah_number

    def test_reference_text_is_ayah_text(
        self, result_1: AlignmentResult, ayah_1: Ayah,
    ) -> None:
        formatted = format_result(result_1)
        assert formatted.reference_text == ayah_1.text

    def test_transcribed_text_preserved(
        self, result_1: AlignmentResult,
    ) -> None:
        formatted = format_result(result_1)
        assert formatted.transcribed_text == result_1.transcribed_text

    def test_rounding_precision_times(
        self, result_1: AlignmentResult,
    ) -> None:
        """start_time, end_time, duration rounded to 3 decimal places."""
        formatted = format_result(result_1)
        assert formatted.start_time == 5.123
        assert formatted.end_time == 8.988
        # Duration computed from already-rounded values for consistency
        assert formatted.duration == round(8.988 - 5.123, 3)

    def test_duration_consistent_with_times(
        self, result_1: AlignmentResult,
    ) -> None:
        """duration must equal end_time - start_time in the output."""
        formatted = format_result(result_1)
        assert formatted.duration == round(
            formatted.end_time - formatted.start_time, 3,
        )

    def test_rounding_precision_similarity(
        self, result_1: AlignmentResult,
    ) -> None:
        """similarity_score rounded to 4 decimal places."""
        formatted = format_result(result_1)
        assert formatted.similarity_score == 0.9512

    def test_computed_fields_match_source(
        self, result_1: AlignmentResult,
    ) -> None:
        formatted = format_result(result_1)
        assert formatted.is_high_confidence == result_1.is_high_confidence
        assert formatted.overlap_detected == result_1.overlap_detected

    def test_overlap_detected_true(
        self, result_2: AlignmentResult,
    ) -> None:
        formatted = format_result(result_2)
        assert formatted.overlap_detected is True

    def test_low_confidence_result(
        self, result_2: AlignmentResult,
    ) -> None:
        """Result with score 0.72 should NOT be high confidence."""
        formatted = format_result(result_2)
        assert formatted.is_high_confidence is False


# --------------- format_results() ---------------


class TestFormatResults:
    """Tests for formatting a list of results into the full envelope."""

    def test_returns_formatted_output(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        assert isinstance(output, FormattedOutput)

    def test_version_field(
        self, result_1: AlignmentResult,
    ) -> None:
        output = format_results([result_1])
        assert output.version == "1.0"

    def test_total_ayahs(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        assert output.total_ayahs == 2

    def test_results_list(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        assert len(output.results) == 2
        assert all(isinstance(r, FormattedAyahResult) for r in output.results)

    def test_created_at_is_iso(
        self, result_1: AlignmentResult,
    ) -> None:
        output = format_results([result_1])
        # Should parse without error
        parsed = datetime.fromisoformat(output.created_at)
        assert parsed.tzinfo is not None  # Must be timezone-aware

    def test_surah_metadata_single_surah(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        """When all results share the same surah, metadata is populated."""
        output = format_results([result_1, result_2])
        assert output.surah_number == 1
        assert output.surah_name == "الفاتحة"

    def test_surah_metadata_mixed_surahs(
        self,
        result_1: AlignmentResult,
        result_other_surah: AlignmentResult,
    ) -> None:
        """When results span multiple surahs, metadata is None."""
        output = format_results([result_1, result_other_surah])
        assert output.surah_number is None
        assert output.surah_name is None

    def test_optional_metadata_provided(
        self, result_1: AlignmentResult,
    ) -> None:
        output = format_results(
            [result_1], audio_file="surah_001.wav", reciter="Badr Al-Turki",
        )
        assert output.audio_file == "surah_001.wav"
        assert output.reciter == "Badr Al-Turki"

    def test_optional_metadata_default_none(
        self, result_1: AlignmentResult,
    ) -> None:
        output = format_results([result_1])
        assert output.audio_file is None
        assert output.reciter is None


# --------------- FormattedSummary ---------------


class TestFormattedSummary:
    """Tests for summary statistics computation."""

    def test_average_similarity(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        expected_avg = round((0.9512 + 0.72) / 2, 4)
        assert output.summary.average_similarity == expected_avg

    def test_high_confidence_count(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        # result_1: 0.95 (high), result_2: 0.72 (not high)
        assert output.summary.high_confidence_count == 1

    def test_high_confidence_percentage(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        assert output.summary.high_confidence_percentage == 50.0

    def test_overlap_count(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        # Only result_2 has overlap_detected=True
        assert output.summary.overlap_count == 1

    def test_total_duration(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        output = format_results([result_1, result_2])
        # Duration computed from rounded times for consistency
        r1_dur = round(round(8.9876543, 3) - round(5.1234567, 3), 3)
        r2_dur = round(round(13.5, 3) - round(9.0, 3), 3)
        assert output.summary.total_duration == round(r1_dur + r2_dur, 3)


# --------------- Empty results ---------------


class TestEmptyResults:
    """Tests for handling empty result lists."""

    def test_empty_results_valid_output(self) -> None:
        output = format_results([])
        assert isinstance(output, FormattedOutput)
        assert output.total_ayahs == 0
        assert output.results == []

    def test_empty_results_summary_zeros(self) -> None:
        output = format_results([])
        assert output.summary.average_similarity == 0.0
        assert output.summary.high_confidence_count == 0
        assert output.summary.high_confidence_percentage == 0.0
        assert output.summary.overlap_count == 0
        assert output.summary.total_duration == 0.0

    def test_empty_results_no_surah_metadata(self) -> None:
        output = format_results([])
        assert output.surah_number is None
        assert output.surah_name is None


# --------------- format_results_list() ---------------


class TestFormatResultsList:
    """Tests for the flat list convenience function."""

    def test_returns_list(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        result = format_results_list([result_1, result_2])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_returns_formatted_ayah_results(
        self, result_1: AlignmentResult,
    ) -> None:
        result = format_results_list([result_1])
        assert all(isinstance(r, FormattedAyahResult) for r in result)

    def test_empty_list(self) -> None:
        assert format_results_list([]) == []


# --------------- format_to_json() ---------------


class TestFormatToJson:
    """Tests for the JSON string convenience function."""

    def test_returns_string(self, result_1: AlignmentResult) -> None:
        result = format_to_json([result_1])
        assert isinstance(result, str)

    def test_valid_json(self, result_1: AlignmentResult) -> None:
        result = format_to_json([result_1])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "version" in parsed
        assert "results" in parsed

    def test_json_round_trip(
        self, result_1: AlignmentResult, result_2: AlignmentResult,
    ) -> None:
        """Serialize to JSON and parse back yields valid FormattedOutput."""
        json_str = format_to_json([result_1, result_2])
        parsed = FormattedOutput.model_validate_json(json_str)
        assert parsed.total_ayahs == 2
        assert len(parsed.results) == 2

    def test_json_contains_arabic_text(
        self, result_1: AlignmentResult,
    ) -> None:
        """JSON should use ensure_ascii=False for Arabic text."""
        result = format_to_json([result_1])
        assert "بسم" in result  # Arabic should be literal, not escaped


# --------------- Public API export ---------------


class TestPublicAPI:
    """Verify formatter is exported from the top-level munajjam package."""

    def test_format_results_importable(self) -> None:
        from munajjam import format_results as fn
        assert callable(fn)

    def test_format_to_json_importable(self) -> None:
        from munajjam import format_to_json as fn
        assert callable(fn)

    def test_formatted_output_importable(self) -> None:
        from munajjam import FormattedOutput as cls
        assert cls is not None

    def test_formatted_ayah_result_importable(self) -> None:
        from munajjam import FormattedAyahResult as cls
        assert cls is not None
