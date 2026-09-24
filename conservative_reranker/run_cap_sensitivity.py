from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

try:
    from .run_experiment import (
        REPO_ROOT,
        SCRIPT_DIR,
        gpu_utilization,
        parse_cpu_affinity,
        run,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from run_experiment import (  # type: ignore
        REPO_ROOT,
        SCRIPT_DIR,
        gpu_utilization,
        parse_cpu_affinity,
        run,
    )


CAP_CONFIGS = (
    ("baseline_256_128_384", 256, 128, 384),
    ("base_512_128_640", 512, 128, 640),
    ("external_256_256_512", 256, 256, 512),
    ("combined_512_256_768", 512, 256, 768),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one complete pool and run the P0 candidate-cap sensitivity grid."
    )
    parser.add_argument("--dataset", default="icews18")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--rules", required=True)
    parser.add_argument("--source", default="external_cameo/CAMEO.Manual.1.1b3.tex")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queries-per-relation", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--tail-fraction", type=float, default=0.35)
    parser.add_argument("--num-processes", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--nice-level", type=int, default=10)
    parser.add_argument("--gpu-index", type=int, default=-1)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Interpreter used for every child stage; it must import torch successfully.",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def train_archive(
    examples: Path,
    report: Path,
    args: argparse.Namespace,
    cpu_affinity: set[int] | None,
) -> None:
    selected_device = args.device
    selected_gpu = args.gpu_index
    if args.device == "auto" and args.gpu_index >= 0:
        utilization = gpu_utilization(args.gpu_index)
        if utilization is not None and utilization > args.max_gpu_utilization:
            selected_device = "cpu"
            selected_gpu = -1
            tqdm.write(
                f"GPU {args.gpu_index} is at {utilization}%; using CPU for this training run."
            )
    run(
        [
            args.python_executable,
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
        ],
        cpu_affinity=cpu_affinity,
        nice_level=args.nice_level,
        gpu_index=selected_gpu,
    )


def main() -> None:
    args = parse_args()
    runtime_check = subprocess.run(
        [
            args.python_executable,
            "-c",
            "import numpy, torch; print(torch.__version__, numpy.__version__)",
        ],
        text=True,
        capture_output=True,
    )
    if runtime_check.returncode != 0:
        detail = (runtime_check.stderr or runtime_check.stdout).strip()
        raise RuntimeError(
            f"Training interpreter cannot import torch: {args.python_executable}\n{detail}"
        )
    tqdm.write(
        f"Training runtime: {args.python_executable} "
        f"({runtime_check.stdout.strip()})"
    )
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive")
    if args.nice_level < 0 or args.nice_level > 19:
        raise ValueError("--nice-level must be between 0 and 19")
    cpu_affinity = parse_cpu_affinity(args.cpu_affinity) if args.cpu_affinity else None
    if cpu_affinity and args.num_processes > len(cpu_affinity):
        raise ValueError("Worker count must not exceed selected CPU affinity count")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    combined_dir = output_root / "combined_512_256_768"

    large_command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_experiment.py"),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--rules",
        args.rules,
        "--source",
        args.source,
        "--output-dir",
        str(combined_dir),
        "--queries-per-relation",
        str(args.queries_per_relation),
        "--tail-fraction",
        str(args.tail_fraction),
        "--num-processes",
        str(args.num_processes),
        "--rule-processes",
        str(args.num_processes),
        "--max-base-candidates",
        "512",
        "--max-external-candidates",
        "256",
        "--max-union-candidates",
        "768",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
        "--nice-level",
        str(args.nice_level),
        "--gpu-index",
        str(args.gpu_index),
        "--max-gpu-utilization",
        str(args.max_gpu_utilization),
        "--seed",
        str(args.seed),
    ]
    if args.cpu_affinity:
        large_command.extend(("--cpu-affinity", args.cpu_affinity))
    if args.max_queries > 0:
        large_command.extend(("--max-queries", str(args.max_queries)))
    if args.rebuild:
        large_command.append("--rebuild")
    run(
        large_command,
        cpu_affinity=cpu_affinity,
        # run_experiment applies the requested niceness to its own workers;
        # do not add it twice through nested subprocesses.
        nice_level=0,
    )

    pool = combined_dir / "train_only_candidates.npz"
    for name, base_cap, external_cap, union_cap in CAP_CONFIGS[:-1]:
        variant_dir = output_root / name
        examples = variant_dir / "train_only_candidates.npz"
        report = variant_dir / "report.json"
        run(
            [
                args.python_executable,
                str(SCRIPT_DIR / "slice_candidates.py"),
                "--source",
                str(pool),
                "--output",
                str(examples),
                "--max-base-candidates",
                str(base_cap),
                "--max-external-candidates",
                str(external_cap),
                "--max-union-candidates",
                str(union_cap),
            ],
            cpu_affinity=cpu_affinity,
            nice_level=args.nice_level,
        )
        train_archive(examples, report, args, cpu_affinity)

    summary = {
        "version": 1,
        "protocol": "p0_candidate_cap_sensitivity",
        "dataset": args.dataset,
        "seed": args.seed,
        "configs": {},
        "validation_read": False,
        "test_read": False,
    }
    for name, base_cap, external_cap, union_cap in CAP_CONFIGS:
        variant_dir = output_root / name
        report = json.loads((variant_dir / "report.json").read_text(encoding="utf-8"))
        metadata = report["examples_metadata"]
        summary["configs"][name] = {
            "max_base_candidates": base_cap,
            "max_external_candidates": external_cap,
            "max_union_candidates": union_cap,
            "candidate_rows": metadata["candidates"],
            "full_archive_base_coverage": metadata["queries_with_base_answer"] / metadata["queries"],
            "full_archive_union_coverage": metadata["queries_with_union_answer"] / metadata["queries"],
            "confirmation_results": report["confirmation_results"],
            "report": str((variant_dir / "report.json").resolve()),
        }
    summary_path = output_root / "cap_sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "configs": list(summary["configs"])}, indent=2))


if __name__ == "__main__":
    main()
