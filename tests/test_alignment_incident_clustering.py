import pytest
from app.core.validator import (
    AlignmentIncident,
    cluster_alignment_findings,
    AlignmentRegion
)

def test_cluster_empty_findings():
    assert cluster_alignment_findings([]) == []
    assert cluster_alignment_findings([], total_cues=100) == []

def test_cluster_single_finding():
    raw = [
        {"start_idx": 10, "end_idx": 15, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Shifted at 11-16"}
    ]
    incidents = cluster_alignment_findings(raw, total_cues=100)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.start_idx == 10
    assert inc.end_idx == 15
    assert inc.verdict == "SHIFT_PLUS_1"
    assert inc.confidence == "HIGH"
    assert inc.confirmation_required is False
    assert len(inc.supporting_findings) == 1

def test_cluster_overlapping_same_verdict():
    raw = [
        {"start_idx": 10, "end_idx": 15, "verdict": "SHIFT_MINUS_1", "confidence": "MEDIUM", "details": "Finding A"},
        {"start_idx": 13, "end_idx": 18, "verdict": "SHIFT_MINUS_1", "confidence": "HIGH", "details": "Finding B"},
    ]
    incidents = cluster_alignment_findings(raw, total_cues=100)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.start_idx == 10
    assert inc.end_idx == 18
    assert inc.verdict == "SHIFT_MINUS_1"
    assert inc.confidence == "HIGH"
    assert inc.confirmation_required is False
    assert len(inc.supporting_findings) == 2

def test_cluster_overlapping_conflicting_shifts():
    # PLUS vs MINUS in overlapping range
    raw = [
        {"start_idx": 20, "end_idx": 25, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Shift forward"},
        {"start_idx": 22, "end_idx": 28, "verdict": "SHIFT_MINUS_1", "confidence": "HIGH", "details": "Shift backward"},
    ]
    incidents = cluster_alignment_findings(raw, total_cues=100)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.start_idx == 20
    assert inc.end_idx == 28
    assert inc.verdict == "CONFLICTING_SHIFT"
    assert inc.confirmation_required is True

def test_cluster_overlapping_complex_shifts():
    # SHIFT vs MERGED in overlapping range
    raw = [
        {"start_idx": 30, "end_idx": 35, "verdict": "SHIFT_PLUS_1", "confidence": "MEDIUM", "details": "Shift"},
        {"start_idx": 33, "end_idx": 40, "verdict": "MERGED", "confidence": "HIGH", "details": "Merged cues"},
    ]
    incidents = cluster_alignment_findings(raw, total_cues=100)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.start_idx == 30
    assert inc.end_idx == 40
    assert inc.verdict == "COMPLEX_SHIFT"
    assert inc.confirmation_required is True

def test_cluster_adjacent_proximity():
    # Adjacent findings within 1 cue (end_idx 10 and start_idx 11) should cluster
    raw = [
        {"start_idx": 5, "end_idx": 10, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Part 1"},
        {"start_idx": 11, "end_idx": 16, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Part 2"},
    ]
    incidents = cluster_alignment_findings(raw, total_cues=100)
    assert len(incidents) == 1
    assert incidents[0].start_idx == 5
    assert incidents[0].end_idx == 16

def test_cluster_distinct_non_overlapping():
    raw = [
        {"start_idx": 5, "end_idx": 10, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "First incident"},
        {"start_idx": 50, "end_idx": 55, "verdict": "SHIFT_MINUS_1", "confidence": "HIGH", "details": "Second incident"},
    ]
    incidents = cluster_alignment_findings(raw, total_cues=100)
    assert len(incidents) == 2
    assert incidents[0].start_idx == 5
    assert incidents[0].end_idx == 10
    assert incidents[0].verdict == "SHIFT_PLUS_1"
    assert incidents[1].start_idx == 50
    assert incidents[1].end_idx == 55
    assert incidents[1].verdict == "SHIFT_MINUS_1"

def test_cluster_bounds_clamping():
    raw = [
        {"start_idx": -5, "end_idx": 150, "verdict": "SHIFT_PLUS_1", "confidence": "HIGH", "details": "Out of bounds"}
    ]
    incidents = cluster_alignment_findings(raw, total_cues=50)
    assert len(incidents) == 1
    assert incidents[0].start_idx == 0
    assert incidents[0].end_idx == 49
