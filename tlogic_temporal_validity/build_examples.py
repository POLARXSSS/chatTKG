from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    Grapher,
    candidate_time_features,
    dataset_dir,
    json_dump,
    load_rules,
    resolve_path,
    store_edges,
    temporal_score,
    unique_queries,
)


DEFAULT_PROCESSES = max(1, (os.cpu_count() or 2) // 2)
ROW_FIELDS = (
    "query_id",
    "query_time",
    "head_relation",
    "candidate",
    "rule_key",
    "delta",
    "recent_count",
    "path_count",
    "tlogic_conf",
    "body_support",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-free pseudo-future examples for temporal-validity "
            "learning."
        )
    )
    parser.add_argument("--dataset", default="icews14")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule-lengths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--min-body-supp", type=int, default=2)
    parser.add_argument(
        "--max-rules-per-relation",
        type=int,
        default=0,
        help="0 keeps every rule that passes the confidence/support filters.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.35,
        help="Fraction of later training timestamps used as pseudo-future queries.",
    )
    parser.add_argument(
        "--queries-per-relation",
        type=int,
        default=64,
        help="Maximum pseudo-future queries sampled for each relation; 0 means all.",
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--frequency-window", type=int, default=30)
    parser.add_argument("--negative-ratio", type=float, default=5.0)
    parser.add_argument("--max-negatives-per-query", type=int, default=50)
    parser.add_argument("--zero-positive-negatives", type=int, default=10)
    parser.add_argument("--hard-negative-decay", type=float, default=0.01)
    parser.add_argument("--num-processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--task-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def resolve_rules_path(path: str, dataset: str) -> Path:
    direct = resolve_path(path)
    if direct.exists():
        return direct
    fallback = resolve_path(Path("output") / dataset / path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Rules file not found: {direct} or {fallback}")


def select_queries(
    train_quads: np.ndarray,
    tail_fraction: float,
    queries_per_relation: int,
    max_queries: int,
    seed: int,
) -> tuple[list[tuple[int, int, int, frozenset[int]]], int]:
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("--tail-fraction must be between 0 and 1")

    queries = unique_queries(train_quads)
    timestamps = sorted({query[2] for query in queries})
    cutoff_index = max(1, int(math.floor(len(timestamps) * (1.0 - tail_fraction))))
    cutoff_index = min(cutoff_index, len(timestamps) - 1)
    cutoff = int(timestamps[cutoff_index])

    by_relation: dict[int, list[tuple[int, int, int, frozenset[int]]]] = (
        defaultdict(list)
    )
    for query in queries:
        if query[2] >= cutoff:
            by_relation[query[1]].append(query)

    rng = np.random.default_rng(seed)
    selected = []
    for relation in sorted(by_relation):
        relation_queries = by_relation[relation]
        order = rng.permutation(len(relation_queries))
        if queries_per_relation > 0:
            order = order[:queries_per_relation]
        selected.extend(relation_queries[int(index)] for index in order)

    if max_queries > 0 and len(selected) > max_queries:
        order = rng.permutation(len(selected))[:max_queries]
        selected = [selected[int(index)] for index in order]
    selected.sort(key=lambda query: (query[2], query[1], query[0]))
    return selected, cutoff


def empty_rows() -> dict[str, list]:
    return {field: [] for field in ROW_FIELDS}


def _worker_init(
    data_path: str,
    rules_path: str,
    selected: list[tuple[int, int, int, frozenset[int]]],
    rule_lengths: list[int],
    min_conf: float,
    min_body_supp: int,
    max_rules_per_relation: int,
    frequency_window: int,
    negative_ratio: float,
    max_negatives_per_query: int,
    zero_positive_negatives: int,
    hard_negative_decay: float,
) -> None:
    global BUILD_DATA, BUILD_RULES, BUILD_SELECTED
    global FREQUENCY_WINDOW, NEGATIVE_RATIO, MAX_NEGATIVES_PER_QUERY
    global ZERO_POSITIVE_NEGATIVES, HARD_NEGATIVE_DECAY
    BUILD_DATA = Grapher(data_path)
    BUILD_RULES = load_rules(
        rules_path,
        rule_lengths=rule_lengths,
        min_conf=min_conf,
        min_body_supp=min_body_supp,
        max_rules_per_relation=max_rules_per_relation,
    )
    BUILD_SELECTED = selected
    FREQUENCY_WINDOW = frequency_window
    NEGATIVE_RATIO = negative_ratio
    MAX_NEGATIVES_PER_QUERY = max_negatives_per_query
    ZERO_POSITIVE_NEGATIVES = zero_positive_negatives
    HARD_NEGATIVE_DECAY = hard_negative_decay


def build_chunk(task: tuple[int, int]) -> tuple[dict[str, list], tuple[int, int, int, int]]:
    start, end = task
    rows = empty_rows()
    positive_rows = 0
    negative_rows = 0
    queries_with_positive_evidence = 0
    queries_with_any_evidence = 0
    current_time = None
    history_edges = None

    for query_id in range(start, end):
        subject, relation, timestamp, answers = BUILD_SELECTED[query_id]
        if timestamp != current_time:
            historical = BUILD_DATA.train_idx[BUILD_DATA.train_idx[:, 3] < timestamp]
            history_edges = store_edges(historical)
            current_time = timestamp

        evidence = []
        for rule in BUILD_RULES.get(relation, []):
            candidates, deltas, recent_counts, path_counts = candidate_time_features(
                rule, history_edges, subject, timestamp, FREQUENCY_WINDOW
            )
            for candidate, delta, recent_count, path_count in zip(
                candidates, deltas, recent_counts, path_counts
            ):
                label = int(int(candidate) in answers)
                hard_score = temporal_score(
                    rule["_smoothed_conf"],
                    HARD_NEGATIVE_DECAY,
                    0.0,
                    int(delta),
                    int(recent_count),
                    FREQUENCY_WINDOW,
                )
                evidence.append(
                    (
                        label,
                        hard_score,
                        int(candidate),
                        rule,
                        int(delta),
                        int(recent_count),
                        int(path_count),
                    )
                )

        if not evidence:
            continue
        queries_with_any_evidence += 1
        positives = [item for item in evidence if item[0] == 1]
        negatives = [item for item in evidence if item[0] == 0]
        if positives:
            queries_with_positive_evidence += 1
            positive_candidates = {item[2] for item in positives}
            negative_budget = int(math.ceil(NEGATIVE_RATIO * len(positive_candidates)))
        else:
            negative_budget = ZERO_POSITIVE_NEGATIVES
        negative_budget = min(MAX_NEGATIVES_PER_QUERY, negative_budget)
        hardness_by_candidate: dict[int, float] = {}
        for item in negatives:
            candidate = item[2]
            hardness_by_candidate[candidate] = max(
                hardness_by_candidate.get(candidate, float("-inf")), item[1]
            )
        selected_negative_candidates = {
            candidate
            for candidate, _ in sorted(
                hardness_by_candidate.items(), key=lambda item: item[1], reverse=True
            )[:negative_budget]
        }
        retained = positives + [
            item for item in negatives if item[2] in selected_negative_candidates
        ]

        for label, _, candidate, rule, delta, recent_count, path_count in retained:
            rows["query_id"].append(query_id)
            rows["query_time"].append(timestamp)
            rows["head_relation"].append(relation)
            rows["candidate"].append(candidate)
            rows["rule_key"].append(rule["_tv_key"])
            rows["delta"].append(delta)
            rows["recent_count"].append(recent_count)
            rows["path_count"].append(path_count)
            rows["tlogic_conf"].append(rule["_smoothed_conf"])
            rows["body_support"].append(int(rule.get("body_supp", 0)))
            rows["label"].append(label)
            if label:
                positive_rows += 1
            else:
                negative_rows += 1

    stats = (
        positive_rows,
        negative_rows,
        queries_with_positive_evidence,
        queries_with_any_evidence,
    )
    return rows, stats


def main() -> None:
    args = parse_args()
    started_at = time.time()
    rules_path = resolve_rules_path(args.rules, args.dataset)
    data_path = dataset_dir(args.dataset, args.data_root)
    destination = resolve_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    data = Grapher(data_path)
    selected, cutoff = select_queries(
        data.train_idx,
        tail_fraction=args.tail_fraction,
        queries_per_relation=args.queries_per_relation,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    rows = empty_rows()
    positive_rows = 0
    negative_rows = 0
    queries_with_positive_evidence = 0
    queries_with_any_evidence = 0
    actual_processes = max(1, min(args.num_processes, len(selected)))
    task_size = args.task_size
    if task_size <= 0:
        task_size = max(1, len(selected) // max(1, actual_processes * 8))
    tasks = [
        (start, min(start + task_size, len(selected)))
        for start in range(0, len(selected), task_size)
    ]
    initializer_args = (
        str(data_path),
        str(rules_path),
        selected,
        args.rule_lengths,
        args.min_conf,
        args.min_body_supp,
        args.max_rules_per_relation,
        args.frequency_window,
        args.negative_ratio,
        args.max_negatives_per_query,
        args.zero_positive_negatives,
        args.hard_negative_decay,
    )
    if actual_processes == 1:
        _worker_init(*initializer_args)
        iterator = (build_chunk(task) for task in tasks)
        progress = tqdm(iterator, total=len(tasks), desc="build examples", unit="chunk")
        for chunk_rows, stats in progress:
            for field in ROW_FIELDS:
                rows[field].extend(chunk_rows[field])
            positive_rows += stats[0]
            negative_rows += stats[1]
            queries_with_positive_evidence += stats[2]
            queries_with_any_evidence += stats[3]
    else:
        with Pool(
            processes=actual_processes,
            initializer=_worker_init,
            initargs=initializer_args,
        ) as pool:
            progress = tqdm(
                pool.imap_unordered(build_chunk, tasks),
                total=len(tasks),
                desc="build examples",
                unit="chunk",
            )
            for chunk_rows, stats in progress:
                for field in ROW_FIELDS:
                    rows[field].extend(chunk_rows[field])
                positive_rows += stats[0]
                negative_rows += stats[1]
                queries_with_positive_evidence += stats[2]
                queries_with_any_evidence += stats[3]

    if not rows["label"]:
        raise RuntimeError(
            "No examples were produced. Increase query/rule limits or check the rules file."
        )

    arrays = {
        "query_id": np.asarray(rows["query_id"], dtype=np.int64),
        "query_time": np.asarray(rows["query_time"], dtype=np.int64),
        "head_relation": np.asarray(rows["head_relation"], dtype=np.int64),
        "candidate": np.asarray(rows["candidate"], dtype=np.int64),
        "rule_key": np.asarray(rows["rule_key"], dtype=np.str_),
        "delta": np.asarray(rows["delta"], dtype=np.float32),
        "recent_count": np.asarray(rows["recent_count"], dtype=np.float32),
        "path_count": np.asarray(rows["path_count"], dtype=np.float32),
        "tlogic_conf": np.asarray(rows["tlogic_conf"], dtype=np.float32),
        "body_support": np.asarray(rows["body_support"], dtype=np.float32),
        "label": np.asarray(rows["label"], dtype=np.float32),
    }
    np.savez_compressed(destination, **arrays)

    metadata = {
        "version": 1,
        "dataset": args.dataset,
        "data_dir": str(data_path),
        "rules": str(rules_path),
        "output": str(destination),
        "seed": args.seed,
        "rule_lengths": args.rule_lengths,
        "min_conf": args.min_conf,
        "min_body_supp": args.min_body_supp,
        "max_rules_per_relation": args.max_rules_per_relation,
        "tail_fraction": args.tail_fraction,
        "cutoff_timestamp": cutoff,
        "selected_queries": len(selected),
        "queries_with_any_evidence": queries_with_any_evidence,
        "queries_with_positive_evidence": queries_with_positive_evidence,
        "rows": len(rows["label"]),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "frequency_window": args.frequency_window,
        "negative_ratio": args.negative_ratio,
        "max_negatives_per_query": args.max_negatives_per_query,
        "zero_positive_negatives": args.zero_positive_negatives,
        "negative_sampling_unit": "candidate",
        "num_processes": actual_processes,
        "task_size": task_size,
        "elapsed_seconds": round(time.time() - started_at, 6),
    }
    json_dump(metadata, destination.with_suffix(".meta.json"))
    print(
        f"Saved {metadata['rows']} examples "
        f"({positive_rows} positive, {negative_rows} negative) to {destination}"
    )


if __name__ == "__main__":
    main()
