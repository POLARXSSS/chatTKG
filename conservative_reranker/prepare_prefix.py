from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a rule-mining dataset from an early prefix of train.txt only."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.35)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tail_fraction < 1.0:
        raise ValueError("--tail-fraction must be between zero and one")
    source = Path(args.data_dir).resolve()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    timestamps = json.loads((source / "ts2id.json").read_text(encoding="utf-8"))
    train_path = source / "train.txt"
    rows = []
    observed_times = set()
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed training row: {line!r}")
            timestamp = int(timestamps[fields[3]])
            rows.append((line if line.endswith("\n") else line + "\n", timestamp))
            observed_times.add(timestamp)
    unique_times = sorted(observed_times)
    if len(unique_times) < 2:
        raise ValueError("training data must contain at least two timestamps")
    cutoff_index = max(1, int(math.floor(len(unique_times) * (1.0 - args.tail_fraction))))
    cutoff_index = min(cutoff_index, len(unique_times) - 1)
    cutoff = int(unique_times[cutoff_index])
    prefix_rows = [line for line, timestamp in rows if timestamp < cutoff]
    if not prefix_rows:
        raise RuntimeError("prefix split contains no training facts")

    for name in ("entity2id.json", "relation2id.json", "ts2id.json"):
        (destination / name).write_bytes((source / name).read_bytes())
    (destination / "train.txt").write_text("".join(prefix_rows), encoding="utf-8")
    # The legacy Grapher expects these paths. They are deliberately empty and
    # the source validation/test files are never opened.
    (destination / "valid.txt").write_text("", encoding="utf-8")
    (destination / "test.txt").write_text("", encoding="utf-8")

    metadata = {
        "version": 1,
        "method": "strict_train_prefix_for_rule_mining",
        "source_data_dir": str(source),
        "output_data_dir": str(destination),
        "source_train_sha256": sha256(train_path),
        "prefix_train_sha256": sha256(destination / "train.txt"),
        "tail_fraction": args.tail_fraction,
        "train_cutoff_timestamp": cutoff,
        "source_train_rows": len(rows),
        "prefix_train_rows": len(prefix_rows),
        "validation_read": False,
        "test_read": False,
    }
    (destination / "prefix.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
