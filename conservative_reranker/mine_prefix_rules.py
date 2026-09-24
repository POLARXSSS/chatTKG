from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MYCODE_DIR = REPO_ROOT / "mycode"
sys.path.insert(0, str(MYCODE_DIR))

from rule_learning import Rule_Learner  # noqa: E402
from temporal_walk import Temporal_Walk  # noqa: E402

try:
    from .train_only_graph import TrainOnlyGrapher
except ImportError:  # pragma: no cover - exercised by CLI execution
    from train_only_graph import TrainOnlyGrapher  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine TLogic rules using only a prepared early training prefix."
    )
    parser.add_argument("--prefix-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule-lengths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num-walks", type=int, default=200)
    parser.add_argument("--transition-distr", default="exp")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mine_worker(
    process_id: int,
    relations: list[int],
    chunk_size: int,
    process_count: int,
    facts: np.ndarray,
    inverse_relations: dict[int, int],
    id2relation: dict[int, str],
    rule_lengths: list[int],
    num_walks: int,
    transition_distribution: str,
    seed: int,
) -> dict[int, list[dict]]:
    np.random.seed(seed)
    walker = Temporal_Walk(facts, inverse_relations, transition_distribution)
    # Initialize the legacy learner without invoking its output-directory side
    # effect; this script owns all artifact paths explicitly.
    learner = Rule_Learner.__new__(Rule_Learner)
    learner.edges = walker.edges
    learner.id2relation = id2relation
    learner.inv_relation_id = inverse_relations
    learner.found_rules = []
    learner.rules_dict = {}
    start = process_id * chunk_size
    end = len(relations) if process_id == process_count - 1 else min(len(relations), start + chunk_size)
    for relation in relations[start:end]:
        for rule_length in rule_lengths:
            for _ in range(num_walks):
                walked, walk = walker.sample_walk(rule_length + 1, relation)
                if walked:
                    learner.create_rule(walk)
    return learner.rules_dict


def merge_rules(parts: list[dict[int, list[dict]]]) -> dict[int, list[dict]]:
    merged: dict[int, list[dict]] = {}
    for part in parts:
        for relation, rules in part.items():
            merged.setdefault(int(relation), []).extend(rules)
    for relation in merged:
        unique = {}
        for rule in merged[relation]:
            signature = json.dumps(
                [rule["head_rel"], rule["body_rels"], rule.get("var_constraints", [])],
                sort_keys=True,
                separators=(",", ":"),
            )
            existing = unique.get(signature)
            if existing is None or float(rule.get("conf", 0.0)) > float(existing.get("conf", 0.0)):
                unique[signature] = rule
        merged[relation] = sorted(
            unique.values(), key=lambda rule: float(rule.get("conf", 0.0)), reverse=True
        )
    return merged


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.num_walks < 1 or args.num_processes < 1:
        raise ValueError("walk and process counts must be positive")
    prefix_dir = Path(args.prefix_dir).resolve()
    output = Path(args.output).resolve()
    prefix_metadata_path = prefix_dir / "prefix.meta.json"
    if not prefix_metadata_path.exists():
        raise FileNotFoundError(f"Missing prefix metadata: {prefix_metadata_path}")
    prefix_metadata = json.loads(prefix_metadata_path.read_text(encoding="utf-8"))
    if prefix_metadata.get("validation_read") is not False or prefix_metadata.get("test_read") is not False:
        raise ValueError("prefix metadata does not certify the leakage boundary")

    data = TrainOnlyGrapher(prefix_dir)
    walker = Temporal_Walk(data.train_idx, data.inv_relation_id, args.transition_distr)
    relations = sorted(int(value) for value in walker.edges)
    processes = min(args.num_processes, len(relations))
    chunk_size = max(1, int(math.ceil(len(relations) / processes)))
    generated_parts = Parallel(n_jobs=processes, return_as="generator")(
        delayed(mine_worker)(
            process_id,
            relations,
            chunk_size,
            processes,
            data.train_idx,
            data.inv_relation_id,
            data.id2relation,
            args.rule_lengths,
            args.num_walks,
            args.transition_distr,
            args.seed,
        )
        for process_id in range(processes)
    )
    parts = list(
        tqdm(
            generated_parts,
            total=processes,
            desc="Rule-mining workers",
            unit="worker",
        )
    )
    rules = merge_rules(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rules), encoding="utf-8")
    metadata = {
        "version": 2,
        "method": "tlogic_rules_mined_from_strict_train_prefix",
        "prefix_dir": str(prefix_dir),
        "prefix_train_sha256": sha256(prefix_dir / "train.txt"),
        "source_train_sha256": prefix_metadata["source_train_sha256"],
        "train_cutoff_timestamp": prefix_metadata["train_cutoff_timestamp"],
        "tail_fraction": prefix_metadata["tail_fraction"],
        "rule_lengths": args.rule_lengths,
        "num_walks": args.num_walks,
        "transition_distribution": args.transition_distr,
        "num_processes": processes,
        "seed": args.seed,
        "relations_with_rules": len(rules),
        "rules": int(sum(len(values) for values in rules.values())),
        "output": str(output),
        "output_sha256": sha256(output),
        "train_only_graph_sha256": sha256((SCRIPT_DIR / "train_only_graph.py").resolve()),
        "elapsed_seconds": time.time() - started,
        "validation_read": False,
        "test_read": False,
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
