from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .features import CANDIDATE_FEATURES, FEATURE_INDEX
except ImportError:  # pragma: no cover - direct CLI execution
    from features import CANDIDATE_FEATURES, FEATURE_INDEX  # type: ignore


BASE_FEATURES = (
    "tlogic_score",
    "tlogic_rank_reciprocal",
    "source_tlogic",
    "rule_evidence_log1p",
    "max_rule_confidence",
    "min_delta_recency",
    "recent_count_log1p",
    "path_count_log1p",
    "body_support_log1p",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_mask(ranks: np.ndarray, limit: int) -> np.ndarray:
    return (ranks > 0) & ((ranks <= limit) if limit > 0 else True)


def slice_arrays(
    arrays: dict[str, np.ndarray],
    max_base: int,
    max_external: int,
    max_union: int,
) -> dict[str, np.ndarray]:
    required_auxiliary = {"candidate_source_rank", "candidate_priority_score"}
    missing = required_auxiliary - set(arrays)
    if missing:
        raise ValueError(
            "exact cap slicing requires auxiliary source ranks and float64 priorities; "
            f"missing={sorted(missing)}. Rebuild the large pool with the current builder."
        )
    candidate_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    query_features: list[np.ndarray] = []
    counts: list[int] = []
    base_present: list[bool] = []
    union_present: list[bool] = []
    source_rank_parts: list[np.ndarray] = []
    priority_score_parts: list[np.ndarray] = []

    offsets = arrays["query_offsets"]
    for query_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        candidates = arrays["candidate"][start:end]
        features = arrays["candidate_features"][start:end].copy()
        labels = arrays["label"][start:end]
        source_ranks = arrays["candidate_source_rank"][start:end].copy()
        raw_priority = arrays["candidate_priority_score"][start:end]

        base_keep = rank_mask(source_ranks[:, 0], max_base)
        exact_keep = rank_mask(source_ranks[:, 1], max_external)
        all_keep = rank_mask(source_ranks[:, 2], max_external)
        official_keep = rank_mask(source_ranks[:, 3], max_external)
        random_keep = rank_mask(source_ranks[:, 4], max_external)

        for name in BASE_FEATURES:
            features[~base_keep, FEATURE_INDEX[name]] = 0.0
        features[~exact_keep, FEATURE_INDEX["exact_history_score"]] = 0.0
        features[~exact_keep, FEATURE_INDEX["source_exact"]] = 0.0
        features[~all_keep, FEATURE_INDEX["all_history_score"]] = 0.0
        features[~all_keep, FEATURE_INDEX["source_all"]] = 0.0
        features[~official_keep, FEATURE_INDEX["official_group_score"]] = 0.0
        features[~random_keep, FEATURE_INDEX["random_group_score"]] = 0.0

        universe = base_keep | all_keep
        priority = np.maximum.reduce(
            [
                np.where(base_keep, raw_priority[:, 0], 0.0),
                np.where(all_keep, raw_priority[:, 1], 0.0),
                np.where(official_keep, raw_priority[:, 2], 0.0),
                np.where(random_keep, raw_priority[:, 3], 0.0),
            ]
        )
        available = np.flatnonzero(universe)
        order = np.lexsort((candidates[available], -priority[available]))
        selected = available[order]
        if max_union > 0:
            selected = selected[:max_union]

        # Recompute base ranks after reducing the base cap.
        selected_base = selected[base_keep[selected]]
        base_order = np.lexsort(
            (
                candidates[selected_base],
                -features[selected_base, FEATURE_INDEX["tlogic_score"]],
            )
        )
        for rank, row in enumerate(selected_base[base_order], start=1):
            features[row, FEATURE_INDEX["tlogic_rank_reciprocal"]] = 1.0 / rank

        kept_candidates = candidates[selected]
        kept_features = features[selected]
        kept_labels = labels[selected]
        kept_source_ranks = source_ranks[selected]
        keep_by_source = np.column_stack(
            (base_keep, exact_keep, all_keep, official_keep, random_keep)
        )[selected]
        kept_source_ranks[~keep_by_source] = 0
        kept_priority = raw_priority[selected].copy()
        kept_priority[~keep_by_source[:, 0], 0] = 0.0
        kept_priority[~keep_by_source[:, 2], 1] = 0.0
        kept_priority[~keep_by_source[:, 3], 2] = 0.0
        kept_priority[~keep_by_source[:, 4], 3] = 0.0
        base_scores = kept_priority[kept_source_ranks[:, 0] > 0, 0]
        base_scores = np.sort(base_scores)[::-1]
        old_query = arrays["query_features"][query_index]
        new_query = old_query.copy()
        new_query[0] = np.log1p(len(selected))
        new_query[1] = np.log1p(len(base_scores))
        new_query[2] = base_scores[0] if len(base_scores) else 0.0
        new_query[3] = (
            base_scores[0] - base_scores[1] if len(base_scores) > 1 else 0.0
        )
        new_query[5] = (
            float(np.sum(kept_features[:, FEATURE_INDEX["source_tlogic"]] < 0.5))
            / max(1, len(selected))
        )

        candidate_parts.append(kept_candidates)
        feature_parts.append(kept_features)
        label_parts.append(kept_labels)
        source_rank_parts.append(kept_source_ranks)
        priority_score_parts.append(kept_priority)
        query_features.append(new_query)
        counts.append(len(selected))
        base_present.append(bool(np.any(labels[base_keep] > 0.5)))
        union_present.append(bool(np.any(kept_labels > 0.5)))

    result = {
        key: value.copy()
        for key, value in arrays.items()
        if key
        not in {
            "candidate",
            "candidate_features",
            "label",
            "query_features",
            "query_offsets",
            "candidate_source_rank",
            "candidate_priority_score",
            "base_answer_present",
            "union_answer_present",
        }
    }
    result.update(
        {
            "candidate": np.concatenate(candidate_parts),
            "candidate_features": np.concatenate(feature_parts, axis=0),
            "candidate_source_rank": np.concatenate(source_rank_parts, axis=0),
            "candidate_priority_score": np.concatenate(priority_score_parts, axis=0),
            "label": np.concatenate(label_parts),
            "query_features": np.asarray(query_features, dtype=np.float32),
            "query_offsets": np.concatenate(
                ([0], np.cumsum(np.asarray(counts, dtype=np.int64)))
            ),
            "base_answer_present": np.asarray(base_present, dtype=np.bool_),
            "union_answer_present": np.asarray(union_present, dtype=np.bool_),
        }
    )
    return result


def derive_candidates(
    source: Path,
    output: Path,
    max_base: int,
    max_external: int,
    max_union: int,
) -> dict:
    metadata_path = source.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_config = metadata["config"]
    source_limits = (
        int(source_config["max_base_candidates"]),
        int(source_config["max_external_candidates"]),
        int(source_config["max_union_candidates"]),
    )
    requested = (max_base, max_external, max_union)
    if any(value < 0 for value in requested):
        raise ValueError("candidate limits must be non-negative")
    for target, available, name in zip(
        requested,
        source_limits,
        ("base", "external", "union"),
    ):
        if available > 0 and (target <= 0 or target > available):
            raise ValueError(f"requested {name} cap {target} exceeds pool cap {available}")
    source_base, source_external, source_union = source_limits
    if source_union > 0 and source_union < source_base + source_external:
        raise ValueError(
            "source pool may have discarded rows needed for exact offline slicing; "
            "build it with max_union >= max_base + max_external"
        )

    with np.load(source) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    derived = slice_arrays(arrays, max_base, max_external, max_union)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **derived)

    updated = dict(metadata)
    updated["version"] = max(3, int(metadata.get("version", 0)))
    updated["method"] = "train_only_conservative_reranker_examples_cap_slice"
    updated["output"] = str(output.resolve())
    updated["parent_examples"] = str(source.resolve())
    updated["parent_examples_sha256"] = sha256(source)
    updated["cap_slice_exact"] = True
    updated["config"] = dict(source_config)
    updated["config"].update(
        {
            "max_base_candidates": max_base,
            "max_external_candidates": max_external,
            "max_union_candidates": max_union,
        }
    )
    updated["candidates"] = int(len(derived["candidate"]))
    updated["queries_with_base_answer"] = int(derived["base_answer_present"].sum())
    updated["queries_with_union_answer"] = int(derived["union_answer_present"].sum())
    updated["positive_candidates"] = int(derived["label"].sum())
    updated["slicer_sha256"] = sha256(Path(__file__).resolve())
    output.with_suffix(".meta.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive an exact smaller-cap archive from a complete large candidate pool."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-base-candidates", type=int, required=True)
    parser.add_argument("--max-external-candidates", type=int, required=True)
    parser.add_argument("--max-union-candidates", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = derive_candidates(
        Path(args.source),
        Path(args.output),
        args.max_base_candidates,
        args.max_external_candidates,
        args.max_union_candidates,
    )
    print(
        json.dumps(
            {
                "output": metadata["output"],
                "config": metadata["config"],
                "queries": metadata["queries"],
                "candidates": metadata["candidates"],
                "queries_with_base_answer": metadata["queries_with_base_answer"],
                "queries_with_union_answer": metadata["queries_with_union_answer"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
