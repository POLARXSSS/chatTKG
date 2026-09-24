from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def available_cpus() -> set[int]:
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))
    return set(range(os.cpu_count() or 1))


def parse_cpu_affinity(specification: str) -> set[int]:
    cpus: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("CPU affinity must select at least one CPU")
    unavailable = cpus - available_cpus()
    if unavailable:
        raise ValueError(f"CPU affinity includes unavailable CPUs: {sorted(unavailable)}")
    return cpus


DEFAULT_PROCESSES = min(32, max(1, len(available_cpus()) // 2))


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
    parser.add_argument("--num-processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--rule-processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--num-walks", type=int, default=200)
    parser.add_argument("--max-base-candidates", type=int, default=256)
    parser.add_argument("--max-external-candidates", type=int, default=128)
    parser.add_argument("--max-union-candidates", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--cpu-affinity",
        default="",
        help="Linux CPU list/ranges inherited by all experiment subprocesses, e.g. 24-47.",
    )
    parser.add_argument(
        "--nice-level",
        type=int,
        default=0,
        help="Positive Unix niceness increment applied only to experiment subprocesses.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=-1,
        help="Physical GPU index exposed to training; -1 keeps the current environment.",
    )
    parser.add_argument(
        "--max-gpu-utilization",
        type=int,
        default=20,
        help="With --device auto, fall back to CPU if the selected GPU is busier than this percent.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--allow-unverified-rules",
        action="store_true",
        help="Smoke-test only: allow a supplied rule file without prefix provenance.",
    )
    return parser.parse_args()


def gpu_utilization(index: int) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def run(
    command: list[str],
    *,
    multiprocess_workers: bool = False,
    cpu_affinity: set[int] | None = None,
    nice_level: int = 0,
    gpu_index: int = -1,
) -> None:
    tqdm.write(" ".join(command))
    environment = os.environ.copy()
    if multiprocess_workers:
        # Prevent each worker from creating its own BLAS/OpenMP thread pool.
        # Parallelism is supplied by --num-processes/--rule-processes instead.
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
    if gpu_index >= 0:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    child_setup = None
    if os.name == "posix" and (cpu_affinity or nice_level > 0):
        def configure_child() -> None:
            if cpu_affinity:
                os.sched_setaffinity(0, cpu_affinity)
            if nice_level > 0:
                os.nice(nice_level)
        child_setup = configure_child

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=environment,
        preexec_fn=child_setup,
    )


def main() -> None:
    args = parse_args()
    if args.nice_level < 0 or args.nice_level > 19:
        raise ValueError("--nice-level must be between 0 and 19")
    if min(
        args.max_base_candidates,
        args.max_external_candidates,
        args.max_union_candidates,
    ) < 0:
        raise ValueError("candidate limits must be non-negative")
    cpu_affinity = parse_cpu_affinity(args.cpu_affinity) if args.cpu_affinity else None
    if cpu_affinity and max(args.num_processes, args.rule_processes) > len(cpu_affinity):
        raise ValueError("Worker count must not exceed the selected CPU affinity count")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = output_dir / "train_only_candidates.npz"
    report = output_dir / "report.json"
    rules = Path(args.rules) if args.rules else output_dir / "prefix_rules.json"

    selected_device = args.device
    selected_gpu = args.gpu_index
    observed_gpu_utilization = None
    if args.gpu_index >= 0:
        observed_gpu_utilization = gpu_utilization(args.gpu_index)
        if (
            args.device == "auto"
            and observed_gpu_utilization is not None
            and observed_gpu_utilization > args.max_gpu_utilization
        ):
            selected_device = "cpu"
            selected_gpu = -1
            tqdm.write(
                f"GPU {args.gpu_index} utilization is {observed_gpu_utilization}%; "
                "falling back to CPU training to avoid contention."
            )
    resource_manifest = {
        "platform": platform.platform(),
        "python": sys.version,
        "available_cpu_count": len(available_cpus()),
        "cpu_affinity": sorted(cpu_affinity) if cpu_affinity else None,
        "nice_level": args.nice_level,
        "candidate_processes": args.num_processes,
        "rule_processes": args.rule_processes,
        "requested_device": args.device,
        "selected_device": selected_device,
        "requested_gpu_index": args.gpu_index,
        "selected_gpu_index": selected_gpu,
        "observed_gpu_utilization_percent": observed_gpu_utilization,
    }
    (output_dir / "run_resources.json").write_text(
        json.dumps(resource_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tqdm.write(json.dumps(resource_manifest, ensure_ascii=False))

    progress = tqdm(total=4, desc="Overall experiment", unit="stage")
    if not args.rules:
        prefix_dir = output_dir / "prefix_dataset"
        prefix_metadata = prefix_dir / "prefix.meta.json"
        progress.set_postfix_str("prepare prefix")
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
                ],
                cpu_affinity=cpu_affinity,
                nice_level=args.nice_level,
            )
        progress.update(1)
        progress.set_postfix_str("mine prefix rules")
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
                ],
                multiprocess_workers=True,
                cpu_affinity=cpu_affinity,
                nice_level=args.nice_level,
            )
        progress.update(1)
    else:
        progress.set_postfix_str("reuse supplied rules")
        progress.update(2)

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
            "max_base_candidates": args.max_base_candidates,
            "max_external_candidates": args.max_external_candidates,
            "max_union_candidates": args.max_union_candidates,
        }
        observed = {
            "dataset": metadata.get("dataset"),
            "rules": metadata.get("rules"),
            "seed": metadata.get("seed"),
            "tail_fraction": metadata.get("selection_config", {}).get("tail_fraction"),
            "queries_per_relation": metadata.get("selection_config", {}).get("queries_per_relation"),
            "max_queries": metadata.get("selection_config", {}).get("max_queries"),
            "max_base_candidates": metadata.get("config", {}).get("max_base_candidates"),
            "max_external_candidates": metadata.get("config", {}).get("max_external_candidates"),
            "max_union_candidates": metadata.get("config", {}).get("max_union_candidates"),
        }
        if observed != requested:
            raise ValueError(
                f"Existing examples do not match requested configuration. "
                f"observed={observed}, requested={requested}. Use --rebuild."
            )

    progress.set_postfix_str("build candidates")
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
            "--max-base-candidates",
            str(args.max_base_candidates),
            "--max-external-candidates",
            str(args.max_external_candidates),
            "--max-union-candidates",
            str(args.max_union_candidates),
            "--seed",
            str(args.seed),
        ]
        if args.max_queries > 0:
            command.extend(("--max-queries", str(args.max_queries)))
        if args.allow_unverified_rules:
            command.append("--allow-unverified-rules")
        run(
            command,
            multiprocess_workers=True,
            cpu_affinity=cpu_affinity,
            nice_level=args.nice_level,
        )
    progress.update(1)

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
            selected_device,
            "--seed",
            str(args.seed),
        ]
    if args.allow_unverified_rules:
        train_command.append("--allow-unverified-examples")
    progress.set_postfix_str("train and evaluate")
    run(
        train_command,
        cpu_affinity=cpu_affinity,
        nice_level=args.nice_level,
        gpu_index=selected_gpu,
    )
    progress.update(1)
    progress.set_postfix_str("complete")
    progress.close()


if __name__ == "__main__":
    main()
