from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MYCODE_DIR = REPO_ROOT / "mycode"
if str(MYCODE_DIR) not in sys.path:
    sys.path.insert(0, str(MYCODE_DIR))

from grapher import Grapher  # noqa: E402
import rule_application as rule_app  # noqa: E402
from temporal_walk import store_edges  # noqa: E402


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (base or REPO_ROOT) / resolved
    return resolved.resolve()


def dataset_dir(dataset: str, data_root: str | Path | None = None) -> Path:
    root = resolve_path(data_root or "data")
    return root / dataset


def output_dir(dataset: str, output_root: str | Path | None = None) -> Path:
    root = resolve_path(output_root or "output_temporal_validity")
    return root / dataset


def rule_signature(rule: dict) -> str:
    """Return a stable identifier independent of JSON rule ordering."""
    constraints = sorted(
        sorted(int(value) for value in constraint)
        for constraint in rule.get("var_constraints", [])
    )
    payload = [
        int(rule["head_rel"]),
        [int(value) for value in rule["body_rels"]],
        constraints,
    ]
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def empirical_confidence(rule: dict) -> float:
    support = rule.get("rule_supp")
    body_support = rule.get("body_supp")
    if support is not None and body_support:
        return float(support) / float(body_support)
    return float(rule.get("tlogic_conf", rule.get("conf", 0.0)))


def smoothed_confidence(rule: dict, prior_strength: float = 2.0) -> float:
    """Shrink sparse empirical confidences toward an uninformative 0.5."""
    support = rule.get("rule_supp")
    body_support = rule.get("body_supp")
    if support is not None and body_support is not None:
        value = (float(support) + 0.5 * prior_strength) / (
            float(body_support) + prior_strength
        )
    else:
        value = empirical_confidence(rule)
    return float(np.clip(value, 1e-5, 1.0 - 1e-5))


def load_rules(
    rules_path: str | Path,
    rule_lengths: Iterable[int],
    min_conf: float,
    min_body_supp: int,
    max_rules_per_relation: int = 0,
) -> dict[int, list[dict]]:
    path = resolve_path(rules_path)
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    rules = {int(key): value for key, value in raw.items()}
    rules = rule_app.filter_rules(
        rules,
        min_conf=min_conf,
        min_body_supp=min_body_supp,
        rule_lengths=list(rule_lengths),
    )
    prepared: dict[int, list[dict]] = {}
    for relation, relation_rules in rules.items():
        copied = []
        for original in relation_rules:
            rule = dict(original)
            rule["_tv_key"] = rule_signature(rule)
            rule["_empirical_conf"] = empirical_confidence(rule)
            rule["_smoothed_conf"] = smoothed_confidence(rule)
            copied.append(rule)
        copied.sort(
            key=lambda item: (
                item["_smoothed_conf"],
                int(item.get("body_supp", 0)),
            ),
            reverse=True,
        )
        if max_rules_per_relation > 0:
            copied = copied[:max_rules_per_relation]
        prepared[int(relation)] = copied
    return prepared


def unique_queries(quads: np.ndarray) -> list[tuple[int, int, int, frozenset[int]]]:
    """Group all correct answers for the same (subject, relation, timestamp)."""
    answers: dict[tuple[int, int, int], set[int]] = {}
    for subject, relation, obj, timestamp in quads:
        key = (int(subject), int(relation), int(timestamp))
        answers.setdefault(key, set()).add(int(obj))
    return [
        (subject, relation, timestamp, frozenset(objects))
        for (subject, relation, timestamp), objects in answers.items()
    ]


def candidate_time_features(
    rule: dict,
    edges: dict[int, np.ndarray],
    query_subject: int,
    query_timestamp: int,
    frequency_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ground one rule and summarize each candidate's temporal evidence.

    The timestamp feature follows the current TLogic implementation and uses
    timestamp_0, the first body edge. Frequency counts distinct timestamp_0
    occurrences instead of raw path multiplicity, which prevents duplicate
    paths from artificially inflating evidence.
    """
    walk_edges = rule_app.match_body_relations(rule, edges, query_subject)
    if not walk_edges or any(len(group) == 0 for group in walk_edges):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty

    entities, timestamps = rule_app.get_walks_arrays(rule, walk_edges)
    if len(entities) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty

    candidates = entities[:, -1].astype(np.int64)
    first_times = timestamps[:, 0].astype(np.int64)
    unique_candidates, inverse = np.unique(candidates, return_inverse=True)

    latest = np.full(len(unique_candidates), -1, dtype=np.int64)
    np.maximum.at(latest, inverse, first_times)
    deltas = np.maximum(1, int(query_timestamp) - latest)
    path_counts = np.bincount(inverse, minlength=len(unique_candidates)).astype(
        np.int64
    )

    candidate_time_pairs = np.unique(
        np.column_stack((inverse, first_times)), axis=0
    )
    if frequency_window > 0:
        recent = candidate_time_pairs[
            candidate_time_pairs[:, 1]
            >= int(query_timestamp) - frequency_window
        ]
    else:
        recent = candidate_time_pairs
    recent_counts = np.bincount(
        recent[:, 0] if len(recent) else np.empty(0, dtype=np.int64),
        minlength=len(unique_candidates),
    ).astype(np.int64)

    return unique_candidates, deltas, recent_counts, path_counts


def temporal_score(
    confidence: float,
    decay: float,
    frequency_weight: float,
    delta: int | float,
    recent_count: int | float,
    frequency_window: int,
) -> float:
    """Compute an interpretable recency-plus-frequency confidence."""
    time_distance = max(0.0, float(delta) - 1.0)
    recency = float(confidence) * math.exp(-float(decay) * time_distance)
    denominator = math.log1p(max(1, int(frequency_window)))
    frequency = math.log1p(max(0.0, float(recent_count))) / denominator
    return float(np.clip(recency + float(frequency_weight) * frequency, 0.0, 1.0))


def top_h_decayed_noisy_or(
    scores: Iterable[float], top_h: int, aggregation_decay: float
) -> float:
    """Aggregate the strongest partially correlated rule scores."""
    ordered = sorted((float(score) for score in scores), reverse=True)
    if top_h > 0:
        ordered = ordered[:top_h]
    if not ordered:
        return 0.0
    adjusted = [
        np.clip(score * (aggregation_decay**index), 0.0, 1.0)
        for index, score in enumerate(ordered)
    ]
    return float(1.0 - np.prod(1.0 - np.asarray(adjusted, dtype=np.float64)))


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def json_dump(payload: object, path: str | Path) -> None:
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


__all__ = [
    "Grapher",
    "REPO_ROOT",
    "candidate_time_features",
    "dataset_dir",
    "empirical_confidence",
    "inverse_softplus",
    "json_dump",
    "load_rules",
    "output_dir",
    "resolve_path",
    "rule_signature",
    "smoothed_confidence",
    "store_edges",
    "temporal_score",
    "top_h_decayed_noisy_or",
    "unique_queries",
]
