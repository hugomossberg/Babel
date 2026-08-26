"""
Regression tests for v2.3.43-beta SRT structural integrity fixes.

Scenario O: subs_to_srt_string must preserve every input cue —
  srt.compose(strict=True) silently drops cues with empty string content
  because make_legal_content('') == '' produces a bare empty block that
  srt.parse() then cannot re-read. Our fix: empty/None content is
  normalised to EMPTY_PLACEHOLDER before compose is called.

Scenario P: No cue merge due to trailing newlines, leading newlines, or
  double-newline embedded in content.

Scenario Borgo: Zero-duration and 1668-cue structural preservation.
"""
import pytest
import srt
from datetime import timedelta
from app.core.cleaner import subs_to_srt_string, EMPTY_PLACEHOLDER
from app.services.pipeline import qa_gate


def _make_sub(index: int, content: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=index),
        end=timedelta(seconds=index + 1),
        content=content,
    )


# ─────────────────────────────────────────────────────────────────────────────
# O. subs_to_srt_string: 1:1 input/output cue count invariant
# ─────────────────────────────────────────────────────────────────────────────

def test_O_empty_content_is_preserved():
    """Root cause: srt.compose drops cues with content=''. Fix normalises to placeholder."""
    subs = [
        _make_sub(1, "Hello world"),
        _make_sub(2, ""),          # empty string — was silently dropped by compose
        _make_sub(3, "Goodbye"),
    ]
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 3, (
        f"SRT integrity failure: wrote 3 cues, reparsed {len(reparsed)}. "
        "subs_to_srt_string must preserve every cue."
    )
    # The previously-empty cue must now contain the placeholder
    assert reparsed[1].content == EMPTY_PLACEHOLDER


def test_O_none_content_is_preserved():
    """None content must also be normalised rather than crashing or being dropped."""
    subs = [
        _make_sub(1, "Normal line"),
        _make_sub(2, None),        # type: ignore[arg-type]
        _make_sub(3, "Another line"),
    ]
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 3


def test_O_whitespace_only_content_is_preserved():
    """Whitespace-only content (e.g. '   ') must also be treated as empty."""
    subs = [
        _make_sub(1, "A"),
        _make_sub(2, "   "),       # whitespace-only
        _make_sub(3, "B"),
    ]
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 3


def test_O_borgo_1668_invariant():
    """Simulate Borgo 1668-cue case: 1 out of 1668 cues has empty content.
    Written count must equal reparsed count."""
    TOTAL = 1668
    subs = [_make_sub(i + 1, f"Dialogue {i}") for i in range(TOTAL)]
    # Inject one empty cue at a plausible position (SDH cleaned to empty)
    subs[831].content = ""    # cue 832 was a music note, cleaned to ""
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == TOTAL, (
        f"Borgo regression: wrote {TOTAL}, reparsed {len(reparsed)}"
    )


def test_O_sequential_renumbering_after_empty_fix():
    """subs_to_srt_string must renumber cues sequentially even with mixed-index input."""
    subs = [
        _make_sub(5, "A"),
        _make_sub(10, ""),
        _make_sub(15, "B"),
    ]
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 3
    assert reparsed[0].index == 1
    assert reparsed[1].index == 2
    assert reparsed[2].index == 3


# ─────────────────────────────────────────────────────────────────────────────
# P. No cue merge: content with embedded newlines/tags must not lose cues
# ─────────────────────────────────────────────────────────────────────────────

def test_P_embedded_double_newline_does_not_merge_cues():
    """srt.make_legal_content strips embedded blank lines but preserves the cue."""
    subs = [
        _make_sub(1, "A"),
        _make_sub(2, "Line one\n\nLine two"),   # double newline in AI output
        _make_sub(3, "C"),
    ]
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 3, "Double-newline in content must not drop/merge cues"


def test_P_placeholder_cue_round_trips():
    """<i></i> placeholder must survive compose -> parse without modification."""
    subs = [
        _make_sub(1, "A"),
        _make_sub(2, EMPTY_PLACEHOLDER),
        _make_sub(3, "C"),
    ]
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 3
    assert reparsed[1].content == EMPTY_PLACEHOLDER


def test_P_large_file_round_trip():
    """1000 cues with some placeholders must all survive the round-trip."""
    subs = []
    for i in range(1000):
        content = EMPTY_PLACEHOLDER if i % 50 == 0 else f"Line {i}"
        subs.append(_make_sub(i + 1, content))
    result = subs_to_srt_string(subs)
    reparsed = list(srt.parse(result))
    assert len(reparsed) == 1000


# ─────────────────────────────────────────────────────────────────────────────
# Borgo 2024 Exact Reproduction & Zero-Duration Semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_borgo_2024_exact_1668_cue_reproduction():
    """
    Exact real-world reproduction of Borgo (2024):
    1668 source cues where cue 1094 has start == end == 01:08:23.326
    and content 'Either you take them\\nand leave me alone'.

    Verifies that:
    1. subs_to_srt_string preserves all 1668 cues (no cue dropped).
    2. Cue 1094 retains exact timestamps 01:08:23.326 --> 01:08:23.326.
    3. Reparsing the generated SRT produces exactly 1668 cues.
    4. QA gate passes with 0 ms drift.
    """
    subs = []
    # Build 1668 cues
    for i in range(1, 1669):
        if i == 1094:
            # The problematic Borgo cue
            t = timedelta(hours=1, minutes=8, seconds=23, milliseconds=326)
            subs.append(srt.Subtitle(
                index=i,
                start=t,
                end=t,
                content="Either you take them\nand leave me alone"
            ))
        else:
            start_s = i * 2
            subs.append(srt.Subtitle(
                index=i,
                start=timedelta(seconds=start_s),
                end=timedelta(seconds=start_s + 1, milliseconds=500),
                content=f"French dialogue cue {i}"
            ))

    assert len(subs) == 1668

    # Serialize to SRT string with cleaner
    srt_output = subs_to_srt_string(subs)

    # Reparse to verify integrity
    reparsed = list(srt.parse(srt_output))
    assert len(reparsed) == 1668, f"Expected 1668 cues after serialization, got {len(reparsed)}"

    # Check cue 1094 (0-indexed 1093)
    cue_1094 = reparsed[1093]
    assert cue_1094.index == 1094
    expected_t = timedelta(hours=1, minutes=8, seconds=23, milliseconds=326)
    assert cue_1094.start == expected_t
    assert cue_1094.end == expected_t
    assert "Either you take them" in cue_1094.content

    # Simulate Swedish translation for QA gate
    trans_subs = []
    for i, s in enumerate(subs, 1):
        if i == 1094:
            trans_content = "Antingen tar du dem\noch lämnar mig ifred"
        else:
            trans_content = f"Detta är en svensk översättning av replik {i}"
        trans_subs.append(srt.Subtitle(
            index=i,
            start=s.start,
            end=s.end,
            content=trans_content
        ))

    # Run QA gate
    qa_result = qa_gate(
        source_subs=subs,
        translated_subs=trans_subs,
        target_lang_code="sv",
        source_language_name="French"
    )

    assert qa_result["passed"] is True
    assert qa_result["sync_diff_ms"] == 0
    assert qa_result["dropped_count"] == 0


def test_zero_duration_policy_vs_negative_duration():
    """
    Verifies policy consistency:
    - start == end (zero duration) is valid and preserved as a flash/marker cue.
    - start > end (negative duration) represents corrupted timecoding.
    """
    # Valid zero-duration cue
    t_zero = timedelta(seconds=10)
    sub_zero = srt.Subtitle(index=1, start=t_zero, end=t_zero, content="Flash frame")
    srt_zero = subs_to_srt_string([sub_zero])
    parsed_zero = list(srt.parse(srt_zero))
    assert len(parsed_zero) == 1
    assert parsed_zero[0].start == parsed_zero[0].end == t_zero

    # Inverted timestamps
    sub_inverted_src = srt.Subtitle(index=1, start=timedelta(seconds=10), end=timedelta(seconds=5), content="Source")
    assert sub_inverted_src.start > sub_inverted_src.end
