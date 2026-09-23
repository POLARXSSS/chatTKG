from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "tlogic_temporal_validity"))

from build_examples import select_queries  # noqa: E402
from common import (  # noqa: E402
    Grapher,
    dataset_dir,
    json_dump,
    load_rules,
    resolve_path,
    store_edges,
)

try:  # Support both direct script execution and package imports.
    from .features import (
        CANDIDATE_FEATURES,
        QUERY_FEATURES,
        HistoryIndex,
        make_feature_rows,
        official_and_random_groups,
        tlogic_candidates,
        truncate_scores,
    )
except ImportError:  # pragma: no cover - exercised by CLI execution
    from features import (  # type: ignore
        CANDIDATE_FEATURES,
        QUERY_FEATURES,
        HistoryIndex,
        make_feature_rows,
        official_and_random_groups,
        tlogic_candidates,
        truncate_scores,
    )


DEFAULT_PROCESSES = max(1, (os.cpu_count() or 2) // 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-free train-only examples for conservative reranking."
    )
    parser.add_argument("--dataset", default="icews14")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--rules", required=True)
    parser.add_argument("--source", default="external_cameo/CAMEO.Manual.1.1b3.tex")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule-lengths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--min-body-supp", type=int, default=2)
    parser.add_argument("--max-rules-per-relation", type=int, default=0)
    parser.add_argument("--tail-fraction", type=float, default=0.35)
    parser.add_argument("--queries-per-relation", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--frequency-window", type=int, default=30)
    parser.add_argument("--history-decay-days", type=float, default=30.0)
    parser.add_argument("--tlogic-decay", type=float, default=0.1)
    parser.add_argument("--tlogic-confidence-weight", type=float, default=0.5)
    parser.add_argument("--max-base-candidates", type=int, default=256)
    parser.add_argument("--max-external-candidates", type=int, default=128)
    parser.add_argument("--max-union-candidates", type=int, default=384)
    parser.add_argument("--num-processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--allow-unverified-rules",
        action="store_true",
        help="Smoke-test only: allow rules without strict-prefix provenance metadata.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_rules_path(path: str, dataset: str) -> Path:
    direct = resolve_path(path)
    if direct.exists():
        return direct
    fallback = resolve_path(Path("output") / dataset / path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Rules file not found: {direct} or {fallback}")


BUILD_DATA: Grapher
BUILD_RULES: dict[int, list[dict]]
BUILD_QUERIES: list[tuple[int, int, int, frozenset[int]]]
BUILD_OFFICIAL: list[str | None]
BUILD_RANDOM: list[str | None]
BUILD_ARGS: dict


def _worker_init(
    data_path: str,
    rules_path: str,
    selected: list[tuple[int, int, int, frozenset[int]]],
    official: list[str | None],
    random_groups: list[str | None],
    config: dict,
) -> None:
    global BUILD_DATA, BUILD_RULES, BUILD_QUERIES
    global BUILD_OFFICIAL, BUILD_RANDOM, BUILD_ARGS
    BUILD_DATA = Grapher(data_path)
    BUILD_RULES = load_rules(
        rules_path,
        rule_lengths=config["rule_lengths"],
        min_conf=config["min_conf"],
        min_body_supp=config["min_body_supp"],
        max_rules_per_relation=config["max_rules_per_relation"],
    )
    BUILD_QUERIES = selected
    BUILD_OFFICIAL = official
    BUILD_RANDOM = random_groups
    BUILD_ARGS = config


def _empty_result() -> dict[str, list]:
    return {
        "candidate": [],
        "candidate_features": [],
        "label": [],
        "query_features": [],
        "query_subject": [],
        "query_relation": [],
        "query_time": [],
        "query_candidate_count": [],
        "base_answer_present_before_truncation": [],
        "base_answer_present": [],
        "union_answer_present": [],
    }


def build_chunk(bounds: tuple[int, int]) -> dict[str, list]:
    start, end = bounds
    result = _empty_result()
    history = HistoryIndex(
        BUILD_DATA.train_idx,
        len(BUILD_DATA.relation2id_original),
        BUILD_OFFICIAL,
        BUILD_RANDOM,
    )
    current_time = None
    edges = None

    for query_index in range(start, end):
        subject, relation, timestamp, answers = BUILD_QUERIES[query_index]
        if timestamp != current_time:
            history.advance_before(int(timestamp))
            historical = BUILD_DATA.train_idx[BUILD_DATA.train_idx[:, 3] < timestamp]
            edges = store_edges(historical)
            current_time = timestamp

        base_all = tlogic_candidates(
            BUILD_RULES.get(int(relation), []),
            edges,
            int(subject),
            int(timestamp),
            BUILD_ARGS["frequency_window"],
            BUILD_ARGS["tlogic_decay"],
            BUILD_ARGS["tlogic_confidence_weight"],
        )
        base_answer_before = any(answer in base_all for answer in answers)
        base = truncate_scores(base_all, BUILD_ARGS["max_base_candidates"])
        exact, all_scores, official, random_scores = history.scores(
            int(subject),
            int(relation),
            int(timestamp),
            BUILD_ARGS["history_decay_days"],
            BUILD_ARGS["max_external_candidates"],
        )
        candidate, candidate_features, labels, query_features = make_feature_rows(
            base,
            exact,
            all_scores,
            official,
            random_scores,
            answers,
            history,
            int(relation),
            BUILD_ARGS["max_union_candidates"],
        )
        result["candidate"].append(candidate)
        result["candidate_features"].append(candidate_features)
        result["label"].append(labels)
        result["query_features"].append(query_features)
        result["query_subject"].append(int(subject))
        result["query_relation"].append(int(relation))
        result["query_time"].append(int(timestamp))
        result["query_candidate_count"].append(len(candidate))
        result["base_answer_present_before_truncation"].append(base_answer_before)
        result["base_answer_present"].append(
            any(answer in base for answer in answers)
        )
        result["union_answer_present"].append(bool(labels.sum() > 0))
    return result


def concatenate_results(parts: list[dict[str, list]]) -> dict[str, np.ndarray]:
    merged = _empty_result()
    for part in parts:
        for key in merged:
            merged[key].extend(part[key])

    counts = np.asarray(merged.pop("query_candidate_count"), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    feature_width = len(CANDIDATE_FEATURES)
    candidates = (
        np.concatenate(merged["candidate"])
        if offsets[-1]
        else np.empty(0, dtype=np.int64)
    )
    candidate_features = (
        np.concatenate(merged["candidate_features"], axis=0)
        if offsets[-1]
        else np.empty((0, feature_width), dtype=np.float32)
    )
    labels = (
        np.concatenate(merged["label"])
        if offsets[-1]
        else np.empty(0, dtype=np.float32)
    )
    return {
        "candidate": candidates.astype(np.int64),
        "candidate_features": candidate_features.astype(np.float32),
        "label": labels.astype(np.float32),
        "query_features": np.asarray(merged["query_features"], dtype=np.float32),
        "query_offsets": offsets,
        "query_subject": np.asarray(merged["query_subject"], dtype=np.int64),
        "query_relation": np.asarray(merged["query_relation"], dtype=np.int64),
        "query_time": np.asarray(merged["query_time"], dtype=np.int64),
        "base_answer_present_before_truncation": np.asarray(
            merged["base_answer_present_before_truncation"], dtype=np.bool_
        ),
        "base_answer_present": np.asarray(merged["base_answer_present"], dtype=np.bool_),
        "union_answer_present": np.asarray(merged["union_answer_present"], dtype=np.bool_),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive")
    if args.history_decay_days <= 0:
        raise ValueError("--history-decay-days must be positive")
    for value in (
        args.max_base_candidates,
        args.max_external_candidates,
        args.max_union_candidates,
    ):
        if value < 0:
            raise ValueError("candidate limits must be non-negative")

    data_path = dataset_dir(args.dataset, args.data_root)
    rules_path = resolve_rules_path(args.rules, args.dataset)
    source_path = resolve_path(args.source)
    destination = resolve_path(args.output)
    data = Grapher(data_path)
    selected, cutoff = select_queries(
        data.train_idx,
        args.tail_fraction,
        args.queries_per_relation,
        args.max_queries,
        args.seed,
    )
    if not selected:
        raise RuntimeError("No pseudo-future queries were selected")

    rules_metadata_path = rules_path.with_suffix(".meta.json")
    rules_metadata = None
    rules_provenance_verified = False
    if rules_metadata_path.exists():
        rules_metadata = json.loads(rules_metadata_path.read_text(encoding="utf-8"))
        checks = {
            "method": rules_metadata.get("method")
            == "tlogic_rules_mined_from_strict_train_prefix",
            "source_train": rules_metadata.get("source_train_sha256")
            == sha256(data_path / "train.txt"),
            "cutoff": int(rules_metadata.get("train_cutoff_timestamp", -1)) == cutoff,
            "rule_hash": rules_metadata.get("output_sha256") == sha256(rules_path),
            "validation_unread": rules_metadata.get("validation_read") is False,
            "test_unread": rules_metadata.get("test_read") is False,
        }
        rules_provenance_verified = all(checks.values())
        if not rules_provenance_verified and not args.allow_unverified_rules:
            raise ValueError(f"Rule provenance checks failed: {checks}")
    elif not args.allow_unverified_rules:
        raise FileNotFoundError(
            f"Missing strict-prefix rule metadata: {rules_metadata_path}. "
            "Mine rules with mine_prefix_rules.py; the override is for smoke tests only."
        )

    official, random_groups, records = official_and_random_groups(
        source_path.read_text(encoding="latin-1"),
        data.relation2id_original,
        args.seed,
    )
    config = {
        "rule_lengths": args.rule_lengths,
        "min_conf": args.min_conf,
        "min_body_supp": args.min_body_supp,
        "max_rules_per_relation": args.max_rules_per_relation,
        "frequency_window": args.frequency_window,
        "history_decay_days": args.history_decay_days,
        "tlogic_decay": args.tlogic_decay,
        "tlogic_confidence_weight": args.tlogic_confidence_weight,
        "max_base_candidates": args.max_base_candidates,
        "max_external_candidates": args.max_external_candidates,
        "max_union_candidates": args.max_union_candidates,
    }
    processes = min(args.num_processes, len(selected))
    chunk_size = int(math.ceil(len(selected) / processes))
    tasks = [
        (start, min(len(selected), start + chunk_size))
        for start in range(0, len(selected), chunk_size)
    ]
    if processes == 1:
        _worker_init(
            str(data_path), str(rules_path), selected, official, random_groups, config
        )
        parts = [build_chunk(task) for task in tasks]
    else:
        with Pool(
            processes,
            initializer=_worker_init,
            initargs=(
                str(data_path),
                str(rules_path),
                selected,
                official,
                random_groups,
                config,
            ),
        ) as pool:
            parts = pool.map(build_chunk, tasks)

    arrays = concatenate_results(parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    num_queries = len(arrays["query_time"])
    metadata = {
        "version": 1,
        "method": "train_only_conservative_reranker_examples",
        "dataset": args.dataset,
        "data_dir": str(data_path),
        "train_sha256": sha256(data_path / "train.txt"),
        "rules": str(rules_path),
        "rules_sha256": sha256(rules_path),
        "rules_metadata": rules_metadata,
        "rules_provenance_verified": rules_provenance_verified,
        "allow_unverified_rules": bool(args.allow_unverified_rules),
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output": str(destination),
        "seed": args.seed,
        "selection_cutoff_timestamp": cutoff,
        "queries": num_queries,
        "candidates": int(len(arrays["candidate"])),
        "queries_with_base_answer_before_truncation": int(
            arrays["base_answer_present_before_truncation"].sum()
        ),
        "queries_with_base_answer": int(arrays["base_answer_present"].sum()),
        "queries_with_union_answer": int(arrays["union_answer_present"].sum()),
        "positive_candidates": int(arrays["label"].sum()),
        "mapped_relations": int(sum(record["code"] is not None for record in records)),
        "total_base_relations": len(records),
        "candidate_feature_names": list(CANDIDATE_FEATURES),
        "query_feature_names": list(QUERY_FEATURES),
        "config": config,
        "selection_config": {
            "tail_fraction": args.tail_fraction,
            "queries_per_relation": args.queries_per_relation,
            "max_queries": args.max_queries,
        },
        "builder_sha256": sha256(Path(__file__).resolve()),
        "features_sha256": sha256((SCRIPT_DIR / "features.py").resolve()),
        "elapsed_seconds": time.time() - started,
        "validation_read": False,
        "test_read": False,
    }
    json_dump(metadata, destination.with_suffix(".meta.json"))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
