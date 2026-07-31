# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for reporting an evaluation result that holds undefined metrics.

The result dicts are hand-built to the shape ``agrieval`` returns, so nothing
here needs embeddings, the dataset, or the hosted API.
"""

import pytest

from precisionai.agriemb import print_evaluation


def _result(**global_metrics: object) -> dict:
    """Build the smallest result dict ``agrieval``'s reporter accepts."""
    return {
        "n_items": 4,
        "embedding_dim": 8,
        "classes": ["Crop | Wheat"],
        "k_values": [5],
        "global_metrics": {
            "pairwise_similarity_stats": {"mean": 0.5, "std": 0.1, "p05": 0.3, "p50": 0.5, "p95": 0.7},
            "centroid_similarity_stats": {
                "mean_cosine_to_centroid": 0.9,
                "std_cosine_to_centroid": 0.05,
                "centroid_norm": 0.8,
            },
            "intra_inter_similarity_gap": {
                "gap": None,
                "mean_intra_class_similarity": None,
                "mean_inter_class_similarity": None,
            },
            "effective_rank": {"effective_rank": None, "effective_rank_ratio": None, "embedding_dim": 8},
            "uniformity": -1.5,
            **global_metrics,
        },
        "per_class": {},
    }


class TestPrintEvaluation:
    """Tests for :func:`precisionai.agriemb.evaluate.print_evaluation`."""

    def test_reports_an_undefined_metric_as_not_available(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_evaluation(_result(alignment=None))

        out = capsys.readouterr().out
        assert "alignment" in out
        assert "n/a" in out
        assert "undefined for this sample" in out

    def test_still_prints_the_rest_of_the_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_evaluation(_result(alignment=None))

        out = capsys.readouterr().out
        assert "n_items      : 4" in out
        assert "uniformity" in out
        assert "pairwise cosine" in out

    def test_formats_a_defined_metric_normally(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_evaluation(_result(alignment=0.25))

        out = capsys.readouterr().out
        assert "alignment          : 0.2500" in out
        assert "n/a" not in out

    def test_does_not_mutate_the_result_it_was_given(self) -> None:
        result = _result(alignment=None)

        print_evaluation(result)

        assert result["global_metrics"]["alignment"] is None
