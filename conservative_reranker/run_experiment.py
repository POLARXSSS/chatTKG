from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the train-only conservative reranker experiment")
    parser.add_argument("--dataset", default="icews14")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--rules",
        default="",
        help="Optional prefix-mined rule file. If omitted, prefix preparation and mining run first.",
    )
    parser.add_argument("--source", default="external_cameo/CAMEO.Manual.1.1b3.tex")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--queries-per-relation", type=int, default=64)
    parser.add_argument("--tail-fraction", type=float, default=0.35)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--rule-processes", type=int, default=1)
    parser.add_argument("--num-walks", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--allow-unverified-rules",
        action="store_true",
        help="Smoke-test only: allow a supplied rule file without prefix provenance.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = output_dir / "train_only_candidates.npz"
    report = output_dir / "report.json"
    rules = Path(args.rules) if args.rules else output_dir / "prefix_rules.json"

    if not args.rules:
        prefix_dir = output_dir / "prefix_dataset"
        prefix_metadata = prefix_dir / "prefix.meta.json"
        if args.rebuild or not prefix_metadata.exists():
            run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "prepare_prefix.py"),
                    "--data-dir",
                    str(REPO_ROOT / args.data_root / args.dataset),
                    "--output-dir",
                    str(prefix_dir),
                    "--tail-fraction",
                    str(args.tail_fraction),
                ]
            )
        if args.rebuild or not rules.exists() or not rules.with_suffix(".meta.json").exists():
            run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "mine_prefix_rules.py"),
                    "--prefix-dir",
                    str(prefix_dir),
                    "--output",
                    str(rules),
                    "--rule-lengths",
                    "1",
                    "2",
                    "3",
                    "--num-walks",
                    str(args.num_walks),
                    "--num-processes",
                    str(args.rule_processes),
                    "--seed",
                    str(args.seed),
                ]
            )

    if examples.exists() and not args.rebuild:
        metadata_path = examples.with_suffix(".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Existing examples lack metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested_rules = rules if rules.is_absolute() else REPO_ROOT / rules
        requested = {
            "dataset": args.dataset,
            "rules": str(requested_rules.resolve()),
            "seed": args.seed,
            "tail_fraction": args.tail_fraction,
            "queries_per_relation": args.queries_per_relation,
            "max_queries": args.max_queries,
        }
        observed = {
            "dataset": metadata.get("dataset"),
            "rules": metadata.get("rules"),
            "seed": metadata.get("seed"),
            "tail_fraction": metadata.get("selection_config", {}).get("tail_fraction"),
            "queries_per_relation": metadata.get("selection_config", {}).get("queries_per_relation"),
            "max_queries": metadata.get("selection_config", {}).get("max_queries"),
        }
        if observed != requested:
            raise ValueError(
                f"Existing examples do not match requested configuration. "
                f"observed={observed}, requested={requested}. Use --rebuild."
            )

    if args.rebuild or not examples.exists():
        command = [
            sys.executable,
            str(SCRIPT_DIR / "build_examples.py"),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--rules",
            str(rules),
            "--source",
            args.source,
            "--output",
            str(examples),
            "--queries-per-relation",
            str(args.queries_per_relation),
            "--tail-fraction",
            str(args.tail_fraction),
            "--num-processes",
            str(args.num_processes),
            "--seed",
            str(args.seed),
        ]
        if args.max_queries > 0:
            command.extend(("--max-queries", str(args.max_queries)))
        if args.allow_unverified_rules:
            command.append("--allow-unverified-rules")
        run(command)

    train_command = [
            sys.executable,
            str(SCRIPT_DIR / "train.py"),
            "--examples",
            str(examples),
            "--output",
            str(report),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
    if args.allow_unverified_rules:
        train_command.append("--allow-unverified-examples")
    run(train_command)


if __name__ == "__main__":
    main()
