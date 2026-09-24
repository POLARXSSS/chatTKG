from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "tlogic_temporal_validity"))
sys.path.insert(0, str(REPO_ROOT / "external_cameo"))

from common import top_h_decayed_noisy_or  # noqa: E402
import rule_application as rule_app  # noqa: E402
from run_study import mapping  # noqa: E402


CANDIDATE_FEATURES = (
    "tlogic_score",
    "tlogic_rank_reciprocal",
    "exact_history_score",
    "all_history_score",
    "official_group_score",
    "random_group_score",
    "source_tlogic",
    "source_exact",
    "source_all",
    "rule_evidence_log1p",
    "max_rule_confidence",
    "min_delta_recency",
    "recent_count_log1p",
    "path_count_log1p",
    "body_support_log1p",
    "candidate_seen_before",
)


MAX_MATERIALIZED_WALK_ROWS = 5_000_000


def _summarize_walks(
    entities: np.ndarray,
    timestamps: np.ndarray,
    query_timestamp: int,
    frequency_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(entities) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    candidates = entities[:, -1].astype(np.int64)
    first_times = timestamps[:, 0].astype(np.int64)
    unique_candidates, inverse = np.unique(candidates, return_inverse=True)
    latest = np.full(len(unique_candidates), -1, dtype=np.int64)
    np.maximum.at(latest, inverse, first_times)
    path_counts = np.bincount(inverse, minlength=len(unique_candidates)).astype(np.int64)
    pairs = np.unique(np.column_stack((inverse, first_times)), axis=0)
    if frequency_window > 0:
        pairs = pairs[pairs[:, 1] >= int(query_timestamp) - frequency_window]
    recent_counts = np.bincount(
        pairs[:, 0] if len(pairs) else np.empty(0, dtype=np.int64),
        minlength=len(unique_candidates),
    ).astype(np.int64)
    return (
        unique_candidates,
        np.maximum(1, int(query_timestamp) - latest),
        recent_counts,
        path_counts,
    )


def _compact_walk_summary(
    rule: dict,
    walk_edges: list[np.ndarray],
    query_timestamp: int,
    frequency_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly aggregate walks without materializing every duplicate path."""
    constraints = rule.get("var_constraints", [])
    track_path = bool(constraints)

    first_counts: dict[tuple, int] = defaultdict(int)
    for source, target, timestamp in np.asarray(walk_edges[0], dtype=np.int64):
        path = (int(source), int(target)) if track_path else int(target)
        first_counts[(path, int(timestamp), int(timestamp))] += 1
    states = first_counts

    for step, raw_edges in enumerate(walk_edges[1:], start=1):
        adjacency: dict[int, dict[tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for source, target, timestamp in np.asarray(raw_edges, dtype=np.int64):
            adjacency[int(source)][(int(target), int(timestamp))] += 1

        next_states: dict[tuple, int] = defaultdict(int)
        for (path, first_time, last_time), multiplicity in states.items():
            current = path[-1] if track_path else path
            for (target, edge_time), edge_multiplicity in adjacency.get(current, {}).items():
                if last_time > edge_time:
                    continue
                next_path = path + (target,) if track_path else target
                if track_path:
                    valid = True
                    for constraint in constraints:
                        known = [index for index in constraint if index <= step + 1]
                        if len(known) > 1 and any(
                            next_path[index] != next_path[known[0]] for index in known[1:]
                        ):
                            valid = False
                            break
                    if not valid:
                        continue
                next_states[(next_path, first_time, edge_time)] += (
                    multiplicity * edge_multiplicity
                )
        states = next_states
        if not states:
            break

    candidate_paths: dict[int, int] = defaultdict(int)
    candidate_first_times: dict[int, set[int]] = defaultdict(set)
    candidate_latest: dict[int, int] = {}
    cutoff = int(query_timestamp) - frequency_window
    for (path, first_time, _), multiplicity in states.items():
        candidate = int(path[-1] if track_path else path)
        candidate_paths[candidate] += multiplicity
        candidate_latest[candidate] = max(candidate_latest.get(candidate, -1), first_time)
        if frequency_window <= 0 or first_time >= cutoff:
            candidate_first_times[candidate].add(first_time)

    candidates = np.asarray(sorted(candidate_paths), dtype=np.int64)
    deltas = np.asarray(
        [max(1, int(query_timestamp) - candidate_latest[value]) for value in candidates],
        dtype=np.int64,
    )
    recent_counts = np.asarray(
        [len(candidate_first_times[value]) for value in candidates], dtype=np.int64
    )
    path_counts = np.asarray(
        [candidate_paths[value] for value in candidates], dtype=np.int64
    )
    return candidates, deltas, recent_counts, path_counts


def safe_candidate_time_features(
    rule: dict,
    edges: dict[int, np.ndarray],
    query_subject: int,
    query_timestamp: int,
    frequency_window: int,
    max_materialized_rows: int = MAX_MATERIALIZED_WALK_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the fast array join unless its next allocation would be excessive."""
    walk_edges = rule_app.match_body_relations(rule, edges, query_subject)
    if not walk_edges or any(len(group) == 0 for group in walk_edges):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty

    first_edges = np.asarray(walk_edges[0])
    entities = first_edges[:, :2]
    timestamps = first_edges[:, 2:3]
    for next_edges in walk_edges[1:]:
        next_edges = np.asarray(next_edges)
        right_subjects = next_edges[:, 0]
        sorted_subjects = np.sort(right_subjects, kind="mergesort")
        starts = np.searchsorted(sorted_subjects, entities[:, -1], side="left")
        ends = np.searchsorted(sorted_subjects, entities[:, -1], side="right")
        if int((ends - starts).sum()) > max_materialized_rows:
            return _compact_walk_summary(
                rule, walk_edges, query_timestamp, frequency_window
            )
        entities, timestamps = rule_app._join_walk_edges(  # noqa: SLF001
            entities, timestamps, next_edges
        )
        if len(entities) == 0:
            break

    if rule.get("var_constraints") and len(entities):
        keep = np.ones(len(entities), dtype=bool)
        for constraint in rule["var_constraints"]:
            for left, right in zip(constraint, constraint[1:]):
                keep &= entities[:, left] == entities[:, right]
        entities = entities[keep]
        timestamps = timestamps[keep]
    return _summarize_walks(
        entities, timestamps, query_timestamp, frequency_window
    )

QUERY_FEATURES = (
    "candidate_count_log1p",
    "tlogic_candidate_count_log1p",
    "tlogic_top_score",
    "tlogic_top_gap",
    "relation_history_log1p",
    "external_only_fraction",
    "inverse_direction",
)

FEATURE_INDEX = {name: index for index, name in enumerate(CANDIDATE_FEATURES)}


def official_and_random_groups(
    source_text: str,
    relations: dict[str, int],
    seed: int,
) -> tuple[list[str | None], list[str | None], list[dict]]:
    records = mapping(source_text, relations)
    official = [record["code"][:2] if record["code"] else None for record in records]
    known = np.flatnonzero([value is not None for value in official])
    random_groups = list(official)
    permutation = np.random.default_rng(seed).permutation(known)
    for left, right in zip(known, permutation):
        random_groups[int(left)] = official[int(right)]
    return official, random_groups, records


def recency_scores(
    history: dict[int, int],
    timestamp: int,
    decay_days: float,
    limit: int = 0,
) -> dict[int, float]:
    values = [
        (int(entity), float(math.exp(-(int(timestamp) - int(last)) / decay_days)))
        for entity, last in history.items()
        if int(last) < int(timestamp)
    ]
    values.sort(key=lambda item: (-item[1], item[0]))
    if limit > 0:
        values = values[:limit]
    return dict(values)


@dataclass
class HistoryIndex:
    facts: np.ndarray
    base_relations: int
    official_groups: list[str | None]
    random_groups: list[str | None]

    def __post_init__(self) -> None:
        order = np.argsort(self.facts[:, 3], kind="stable")
        self.facts = self.facts[order]
        self.position = 0
        self.exact: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        self.all_history: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        self.official: dict[tuple[int, str, int], dict[int, int]] = defaultdict(dict)
        self.random: dict[tuple[int, str, int], dict[int, int]] = defaultdict(dict)
        self.seen_entities: set[int] = set()
        self.relation_counts: dict[int, int] = defaultdict(int)

    @staticmethod
    def _update(store: dict, key: tuple, entity: int, timestamp: int) -> None:
        previous = store[key].get(entity, -1)
        if timestamp > previous:
            store[key][entity] = timestamp

    def advance_before(self, timestamp: int) -> None:
        while self.position < len(self.facts) and int(self.facts[self.position, 3]) < timestamp:
            subject, relation, obj, event_time = map(int, self.facts[self.position])
            direction, base_relation = divmod(relation, self.base_relations)
            self._update(self.exact, (subject, relation), obj, event_time)
            self._update(self.all_history, (subject, direction), obj, event_time)
            official = self.official_groups[base_relation]
            random_group = self.random_groups[base_relation]
            if official is not None:
                self._update(self.official, (subject, official, direction), obj, event_time)
            if random_group is not None:
                self._update(self.random, (subject, random_group, direction), obj, event_time)
            self.seen_entities.add(subject)
            self.seen_entities.add(obj)
            self.relation_counts[relation] += 1
            self.position += 1

    def scores(
        self,
        subject: int,
        relation: int,
        timestamp: int,
        decay_days: float,
        limit: int,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
        direction, base_relation = divmod(int(relation), self.base_relations)
        exact = recency_scores(
            self.exact[(subject, relation)], timestamp, decay_days, limit
        )
        all_scores = recency_scores(
            self.all_history[(subject, direction)], timestamp, decay_days, limit
        )
        official_group = self.official_groups[base_relation]
        random_group = self.random_groups[base_relation]
        official = (
            recency_scores(
                self.official[(subject, official_group, direction)],
                timestamp,
                decay_days,
                limit,
            )
            if official_group is not None
            else {}
        )
        random_scores = (
            recency_scores(
                self.random[(subject, random_group, direction)],
                timestamp,
                decay_days,
                limit,
            )
            if random_group is not None
            else {}
        )
        return exact, all_scores, official, random_scores


def tlogic_candidates(
    relation_rules: Iterable[dict],
    edges: dict[int, np.ndarray],
    subject: int,
    timestamp: int,
    frequency_window: int,
    decay: float,
    confidence_weight: float,
) -> dict[int, dict[str, float]]:
    evidence: dict[int, list[tuple[float, float, int, int, int, int]]] = defaultdict(list)
    for rule in relation_rules:
        candidates, deltas, recent_counts, path_counts = safe_candidate_time_features(
            rule, edges, subject, timestamp, frequency_window
        )
        confidence = float(rule["_empirical_conf"])
        body_support = int(rule.get("body_supp", 0))
        for candidate, delta, recent_count, path_count in zip(
            candidates, deltas, recent_counts, path_counts
        ):
            time_score = math.exp(-float(decay) * max(0.0, float(delta)))
            score = float(
                np.clip(
                    confidence_weight * confidence
                    + (1.0 - confidence_weight) * time_score,
                    0.0,
                    1.0,
                )
            )
            evidence[int(candidate)].append(
                (score, confidence, int(delta), int(recent_count), int(path_count), body_support)
            )

    result: dict[int, dict[str, float]] = {}
    for candidate, values in evidence.items():
        scores = [value[0] for value in values]
        result[candidate] = {
            "score": top_h_decayed_noisy_or(scores, 0, 1.0),
            "evidence_count": float(len(values)),
            "max_confidence": max(value[1] for value in values),
            "min_delta": float(min(value[2] for value in values)),
            "max_recent_count": float(max(value[3] for value in values)),
            "max_path_count": float(max(value[4] for value in values)),
            "max_body_support": float(max(value[5] for value in values)),
        }
    return result


def truncate_scores(values: dict[int, dict[str, float]], limit: int) -> dict[int, dict[str, float]]:
    if limit <= 0 or len(values) <= limit:
        return values
    ordered = sorted(values.items(), key=lambda item: (-item[1]["score"], item[0]))
    return dict(ordered[:limit])


def make_feature_rows(
    base: dict[int, dict[str, float]],
    exact: dict[int, float],
    all_scores: dict[int, float],
    official: dict[int, float],
    random_scores: dict[int, float],
    answers: frozenset[int],
    history: HistoryIndex,
    relation: int,
    max_union_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    universe = set(base) | set(all_scores)
    priority = {
        entity: max(
            base.get(entity, {}).get("score", 0.0),
            all_scores.get(entity, 0.0),
            official.get(entity, 0.0),
            random_scores.get(entity, 0.0),
        )
        for entity in universe
    }
    ordered = sorted(universe, key=lambda entity: (-priority[entity], entity))
    if max_union_candidates > 0:
        ordered = ordered[:max_union_candidates]

    base_order = sorted(base, key=lambda entity: (-base[entity]["score"], entity))
    base_rank = {entity: rank + 1 for rank, entity in enumerate(base_order)}
    top_score = base[base_order[0]]["score"] if base_order else 0.0
    second_score = base[base_order[1]]["score"] if len(base_order) > 1 else 0.0
    query_features = np.asarray(
        [
            math.log1p(len(ordered)),
            math.log1p(len(base)),
            top_score,
            top_score - second_score,
            math.log1p(history.relation_counts.get(int(relation), 0)),
            sum(entity not in base for entity in ordered) / max(1, len(ordered)),
            float(int(relation) >= history.base_relations),
        ],
        dtype=np.float32,
    )

    rows = []
    labels = []
    for entity in ordered:
        item = base.get(entity)
        min_delta = item["min_delta"] if item else 0.0
        rows.append(
            [
                item["score"] if item else 0.0,
                1.0 / base_rank[entity] if entity in base_rank else 0.0,
                exact.get(entity, 0.0),
                all_scores.get(entity, 0.0),
                official.get(entity, 0.0),
                random_scores.get(entity, 0.0),
                float(entity in base),
                float(entity in exact),
                float(entity in all_scores),
                math.log1p(item["evidence_count"]) if item else 0.0,
                item["max_confidence"] if item else 0.0,
                math.exp(-0.1 * min_delta) if item else 0.0,
                math.log1p(item["max_recent_count"]) if item else 0.0,
                math.log1p(item["max_path_count"]) if item else 0.0,
                math.log1p(item["max_body_support"]) if item else 0.0,
                float(entity in history.seen_entities),
            ]
        )
        labels.append(float(entity in answers))
    return (
        np.asarray(ordered, dtype=np.int64),
        np.asarray(rows, dtype=np.float32).reshape(-1, len(CANDIDATE_FEATURES)),
        np.asarray(labels, dtype=np.float32),
        query_features,
    )
