from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import torch

from conservative_reranker.features import (
    CANDIDATE_FEATURES,
    FEATURE_INDEX,
    HistoryIndex,
    make_feature_rows,
    safe_candidate_time_features,
)
from conservative_reranker.model import ConservativeReranker, pairwise_ranking_loss
from conservative_reranker.train import chronological_split, fixed_scores, ranks_from_scores
from conservative_reranker.train_only_graph import TrainOnlyGrapher
from conservative_reranker.run_experiment import parse_cpu_affinity
from conservative_reranker.slice_candidates import slice_arrays


class ConservativeRerankerTests(unittest.TestCase):
    def test_large_candidate_pool_can_be_sliced_to_smaller_caps(self) -> None:
        features = np.zeros((4, len(CANDIDATE_FEATURES)), dtype=np.float32)
        features[:2, FEATURE_INDEX["source_tlogic"]] = 1.0
        features[:2, FEATURE_INDEX["tlogic_score"]] = [0.9, 0.8]
        features[:2, FEATURE_INDEX["tlogic_rank_reciprocal"]] = [1.0, 0.5]
        features[1:, FEATURE_INDEX["source_all"]] = 1.0
        features[1:, FEATURE_INDEX["all_history_score"]] = [0.7, 0.6, 0.5]
        arrays = {
            "candidate": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "candidate_features": features,
            "candidate_source_rank": np.asarray(
                [[1, 0, 0, 0, 0], [2, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 3, 0, 0]],
                dtype=np.int32,
            ),
            "candidate_priority_score": np.asarray(
                [[0.9, 0.0, 0.0, 0.0], [0.8, 0.7, 0.0, 0.0], [0.0, 0.6, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "label": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "query_features": np.zeros((1, 7), dtype=np.float32),
            "query_offsets": np.asarray([0, 4], dtype=np.int64),
            "query_subject": np.asarray([0], dtype=np.int64),
            "query_relation": np.asarray([0], dtype=np.int64),
            "query_time": np.asarray([1], dtype=np.int64),
            "base_answer_present_before_truncation": np.asarray([False]),
            "base_answer_present": np.asarray([False]),
            "union_answer_present": np.asarray([True]),
        }
        sliced = slice_arrays(arrays, max_base=1, max_external=2, max_union=3)
        self.assertEqual(sliced["candidate"].tolist(), [1, 2, 3])
        self.assertEqual(sliced["query_offsets"].tolist(), [0, 3])
        self.assertEqual(
            float(sliced["candidate_features"][1, FEATURE_INDEX["source_tlogic"]]),
            0.0,
        )
        self.assertFalse(bool(sliced["union_answer_present"][0]))

    def test_cpu_affinity_parser(self) -> None:
        available = sorted(__import__("conservative_reranker.run_experiment", fromlist=["available_cpus"]).available_cpus())
        first = available[0]
        second = available[1] if len(available) > 1 else first
        specification = str(first) if first == second else f"{first}-{second}"
        self.assertEqual(parse_cpu_affinity(specification), set(range(first, second + 1)))

    def test_train_only_grapher_does_not_require_validation_or_test(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "entity2id.json").write_text(json.dumps({"a": 0, "b": 1}))
            (root / "relation2id.json").write_text(json.dumps({"r": 0}))
            (root / "ts2id.json").write_text(json.dumps({"1": 0}))
            (root / "train.txt").write_text("a\tr\tb\t1\n")
            graph = TrainOnlyGrapher(root)
            self.assertEqual(graph.train_idx.tolist(), [[0, 0, 1, 0], [1, 1, 0, 0]])
            self.assertFalse((root / "valid.txt").exists())
            self.assertFalse((root / "test.txt").exists())

    def test_history_uses_strictly_earlier_facts(self) -> None:
        facts = np.asarray(
            [
                [0, 0, 1, 1],
                [0, 0, 2, 2],
            ],
            dtype=np.int64,
        )
        history = HistoryIndex(facts, 1, ["01"], ["01"])
        history.advance_before(2)
        exact, all_scores, official, random_scores = history.scores(0, 0, 2, 30.0, 0)
        self.assertIn(1, exact)
        self.assertNotIn(2, exact)
        self.assertEqual(set(all_scores), {1})
        self.assertEqual(set(official), {1})
        self.assertEqual(set(random_scores), {1})

    def test_candidate_truncation_is_label_blind(self) -> None:
        history = HistoryIndex(np.empty((0, 4), dtype=np.int64), 1, ["01"], ["01"])
        base = {
            1: {"score": 0.9, "evidence_count": 1, "max_confidence": 0.8,
                "min_delta": 1, "max_recent_count": 1, "max_path_count": 1,
                "max_body_support": 2},
        }
        candidate, _, labels, _ = make_feature_rows(
            base, {}, {2: 0.8, 3: 0.1}, {}, {}, frozenset({3}), history, 0, 2
        )
        self.assertEqual(candidate.tolist(), [1, 2])
        self.assertEqual(labels.sum(), 0.0)

    def test_compact_walk_summary_matches_materialized_join(self) -> None:
        edges = {
            0: np.asarray([[0, 0, 1, 1], [0, 0, 1, 2]], dtype=np.int64),
            1: np.asarray(
                [[1, 1, 2, 2], [1, 1, 2, 3], [1, 1, 3, 2]], dtype=np.int64
            ),
            2: np.asarray([[2, 2, 4, 3], [3, 2, 4, 4]], dtype=np.int64),
        }
        rule = {"body_rels": [0, 1, 2], "var_constraints": []}
        materialized = safe_candidate_time_features(rule, edges, 0, 6, 3, 1_000_000)
        compact = safe_candidate_time_features(rule, edges, 0, 6, 3, 0)
        for observed, expected in zip(compact, materialized):
            np.testing.assert_array_equal(observed, expected)

    def test_initial_model_preserves_base_scores(self) -> None:
        model = ConservativeReranker(len(CANDIDATE_FEATURES), 7, hidden_size=8)
        features = torch.randn(2, 3, len(CANDIDATE_FEATURES))
        query = torch.randn(2, 7)
        base = torch.rand(2, 3)
        scores, _, delta = model(features, query, base)
        self.assertTrue(torch.allclose(scores, base))
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))

    def test_pairwise_loss_rewards_correct_order(self) -> None:
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        good, _ = pairwise_ranking_loss(
            torch.tensor([[0.9, 0.2, 0.1]]), labels, mask, 0.05, 2
        )
        bad, _ = pairwise_ranking_loss(
            torch.tensor([[0.1, 0.9, 0.8]]), labels, mask, 0.05, 2
        )
        self.assertLess(float(good), float(bad))

    def test_fixed_official_formula(self) -> None:
        raw = np.zeros((1, len(CANDIDATE_FEATURES)), dtype=np.float32)
        raw[0, FEATURE_INDEX["tlogic_score"]] = 0.6
        raw[0, FEATURE_INDEX["all_history_score"]] = 0.4
        raw[0, FEATURE_INDEX["official_group_score"]] = 0.8
        scores, allowed = fixed_scores(raw, "fixed_official")
        expected = 0.8 * (0.95 * 0.6 + 0.05 * 0.4) + 0.2 * 0.8
        self.assertAlmostEqual(float(scores[0]), expected, places=6)
        self.assertTrue(bool(allowed[0]))

    def test_average_tie_rank(self) -> None:
        rank, covered, top1 = ranks_from_scores(
            np.asarray([0.5, 0.5, 0.1]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([True, True, True]),
            10,
        )
        self.assertEqual(rank, 1.5)
        self.assertTrue(covered)
        self.assertFalse(top1)

    def test_chronological_split_is_ordered(self) -> None:
        times = np.repeat(np.arange(10), 2)
        train, selection, confirmation, cutoffs = chronological_split(times, 0.2, 0.2)
        self.assertLess(times[train].max(), times[selection].min())
        self.assertLess(times[selection].max(), times[confirmation].min())
        self.assertEqual(cutoffs["selection_start"], 6)
        self.assertEqual(cutoffs["confirmation_start"], 8)


if __name__ == "__main__":
    unittest.main()
