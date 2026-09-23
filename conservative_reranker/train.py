from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "tlogic_temporal_validity"))

from common import json_dump, resolve_path  # noqa: E402
try:  # Support both direct script execution and package imports.
    from .features import CANDIDATE_FEATURES, FEATURE_INDEX, QUERY_FEATURES
    from .model import ConservativeReranker, pairwise_ranking_loss, preservation_loss
except ImportError:  # pragma: no cover - exercised by CLI execution
    from features import CANDIDATE_FEATURES, FEATURE_INDEX, QUERY_FEATURES  # type: ignore
    from model import (  # type: ignore
        ConservativeReranker,
        pairwise_ranking_loss,
        preservation_loss,
    )


LEARNED_VARIANTS = ("learned_base", "learned_all", "learned_official", "learned_random")
FIXED_METHODS = ("tlogic", "fixed_all", "fixed_official", "fixed_random")
ALL_METHODS = FIXED_METHODS + LEARNED_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate conservative rerankers on train-only time splits."
    )
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output", required=True, help="Destination report JSON")
    parser.add_argument("--variants", nargs="+", choices=LEARNED_VARIANTS, default=list(LEARNED_VARIANTS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--hard-negatives", type=int, default=10)
    parser.add_argument("--max-residual", type=float, default=0.5)
    parser.add_argument("--preservation-weight", type=float, default=0.25)
    parser.add_argument("--gate-weight", type=float, default=0.01)
    parser.add_argument("--distillation-temperature", type=float, default=0.25)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--confirmation-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-width", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--allow-unverified-examples",
        action="store_true",
        help="Smoke-test only: train examples whose rule provenance is not verified.",
    )
    return parser.parse_args()


@dataclass
class Standardizer:
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray
    query_mean: np.ndarray
    query_scale: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def validate_arrays(arrays: dict[str, np.ndarray], metadata: dict) -> None:
    required = {
        "candidate",
        "candidate_features",
        "label",
        "query_features",
        "query_offsets",
        "query_relation",
        "query_time",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Examples are missing arrays: {missing}")
    offsets = arrays["query_offsets"]
    if offsets.ndim != 1 or len(offsets) != len(arrays["query_time"]) + 1:
        raise ValueError("query_offsets must contain one more entry than queries")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("query_offsets must be monotonic and start at zero")
    if offsets[-1] != len(arrays["candidate"]):
        raise ValueError("query_offsets do not cover candidate rows")
    if arrays["candidate_features"].shape != (
        len(arrays["candidate"]),
        len(CANDIDATE_FEATURES),
    ):
        raise ValueError("candidate feature shape does not match the data dictionary")
    if arrays["query_features"].shape != (
        len(arrays["query_time"]),
        len(QUERY_FEATURES),
    ):
        raise ValueError("query feature shape does not match the data dictionary")
    if metadata.get("validation_read") is not False or metadata.get("test_read") is not False:
        raise ValueError("examples do not certify the train-only leakage boundary")


def chronological_split(
    times: np.ndarray,
    selection_fraction: float,
    confirmation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if selection_fraction <= 0 or confirmation_fraction <= 0:
        raise ValueError("selection and confirmation fractions must be positive")
    if selection_fraction + confirmation_fraction >= 1:
        raise ValueError("selection + confirmation fractions must be below one")
    unique = np.unique(times)
    if len(unique) < 5:
        raise ValueError("at least five unique timestamps are required")
    selection_index = int(math.floor(len(unique) * (1.0 - selection_fraction - confirmation_fraction)))
    confirmation_index = int(math.floor(len(unique) * (1.0 - confirmation_fraction)))
    selection_index = min(max(1, selection_index), len(unique) - 2)
    confirmation_index = min(max(selection_index + 1, confirmation_index), len(unique) - 1)
    selection_start = int(unique[selection_index])
    confirmation_start = int(unique[confirmation_index])
    train = np.flatnonzero(times < selection_start)
    selection = np.flatnonzero((times >= selection_start) & (times < confirmation_start))
    confirmation = np.flatnonzero(times >= confirmation_start)
    if min(len(train), len(selection), len(confirmation)) == 0:
        raise ValueError("chronological split produced an empty partition")
    return train, selection, confirmation, {
        "selection_start": selection_start,
        "confirmation_start": confirmation_start,
    }


def candidate_mask(raw: np.ndarray, variant: str) -> np.ndarray:
    source_base = raw[:, FEATURE_INDEX["source_tlogic"]] > 0.5
    if variant in ("tlogic", "learned_base"):
        return source_base
    return np.ones(len(raw), dtype=np.bool_)


def feature_view(raw: np.ndarray, variant: str) -> np.ndarray:
    viewed = raw.copy()
    zero_names: tuple[str, ...]
    if variant == "learned_base":
        zero_names = (
            "exact_history_score",
            "all_history_score",
            "official_group_score",
            "random_group_score",
            "source_exact",
            "source_all",
        )
    elif variant == "learned_all":
        zero_names = ("official_group_score", "random_group_score")
    elif variant == "learned_official":
        zero_names = ("random_group_score",)
    elif variant == "learned_random":
        zero_names = ("official_group_score",)
    else:
        raise ValueError(f"Unknown learned variant: {variant}")
    for name in zero_names:
        viewed[:, FEATURE_INDEX[name]] = 0.0
    return viewed


def query_feature_view(raw: np.ndarray, variant: str) -> np.ndarray:
    viewed = raw.copy()
    if variant == "learned_base":
        viewed[:, QUERY_FEATURES.index("external_only_fraction")] = 0.0
    return viewed


def rows_for_queries(offsets: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
    pieces = [np.arange(offsets[q], offsets[q + 1], dtype=np.int64) for q in query_indices]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def fit_standardizer(
    arrays: dict[str, np.ndarray], train_queries: np.ndarray, variant: str
) -> Standardizer:
    offsets = arrays["query_offsets"]
    rows = rows_for_queries(offsets, train_queries)
    raw = arrays["candidate_features"][rows]
    allowed = candidate_mask(raw, variant)
    viewed = feature_view(raw[allowed], variant)
    if len(viewed) == 0:
        raise RuntimeError(f"{variant} has no training candidates")
    candidate_mean = viewed.mean(axis=0)
    candidate_scale = viewed.std(axis=0)
    candidate_scale[candidate_scale < 1e-6] = 1.0
    queries = query_feature_view(arrays["query_features"][train_queries], variant)
    query_mean = queries.mean(axis=0)
    query_scale = queries.std(axis=0)
    query_scale[query_scale < 1e-6] = 1.0
    return Standardizer(candidate_mean, candidate_scale, query_mean, query_scale)


def make_batch(
    arrays: dict[str, np.ndarray],
    query_indices: np.ndarray,
    variant: str,
    standardizer: Standardizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = arrays["query_offsets"]
    candidate_parts = []
    label_parts = []
    base_parts = []
    source_base_parts = []
    for query in query_indices:
        start, end = int(offsets[query]), int(offsets[query + 1])
        raw = arrays["candidate_features"][start:end]
        allowed = candidate_mask(raw, variant)
        selected = feature_view(raw[allowed], variant)
        candidate_parts.append(selected)
        label_parts.append(arrays["label"][start:end][allowed])
        base_parts.append(raw[allowed, FEATURE_INDEX["tlogic_score"]])
        source_base_parts.append(raw[allowed, FEATURE_INDEX["source_tlogic"]] > 0.5)
    width = max(1, max((len(part) for part in candidate_parts), default=0))
    batch = len(query_indices)
    features = np.zeros((batch, width, len(CANDIDATE_FEATURES)), dtype=np.float32)
    labels = np.zeros((batch, width), dtype=np.float32)
    base_scores = np.zeros((batch, width), dtype=np.float32)
    mask = np.zeros((batch, width), dtype=np.bool_)
    source_base = np.zeros((batch, width), dtype=np.bool_)
    for row, (part, part_labels, part_base, part_source) in enumerate(
        zip(candidate_parts, label_parts, base_parts, source_base_parts)
    ):
        count = len(part)
        if count == 0:
            continue
        features[row, :count] = (part - standardizer.candidate_mean) / standardizer.candidate_scale
        labels[row, :count] = part_labels
        base_scores[row, :count] = part_base
        mask[row, :count] = True
        source_base[row, :count] = part_source
    query = query_feature_view(arrays["query_features"][query_indices], variant)
    query = (query - standardizer.query_mean) / standardizer.query_scale
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (features, query.astype(np.float32), labels, base_scores, mask, source_base)
    )


def fixed_scores(raw: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    base = raw[:, FEATURE_INDEX["tlogic_score"]]
    all_score = raw[:, FEATURE_INDEX["all_history_score"]]
    official = raw[:, FEATURE_INDEX["official_group_score"]]
    random_score = raw[:, FEATURE_INDEX["random_group_score"]]
    if method == "tlogic":
        return base, candidate_mask(raw, method)
    baseline = 0.95 * base + 0.05 * all_score
    if method == "fixed_all":
        return baseline, np.ones(len(raw), dtype=np.bool_)
    if method == "fixed_official":
        return 0.80 * baseline + 0.20 * official, np.ones(len(raw), dtype=np.bool_)
    if method == "fixed_random":
        return 0.80 * baseline + 0.20 * random_score, np.ones(len(raw), dtype=np.bool_)
    raise ValueError(f"Unknown fixed method: {method}")


def ranks_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    allowed: np.ndarray,
    num_entities: int,
) -> tuple[float, bool, bool]:
    scores = scores[allowed]
    labels = labels[allowed] > 0.5
    if len(scores) == 0 or not labels.any():
        return float(num_entities), False, False
    best_positive = float(scores[labels].max())
    negatives = scores[~labels]
    better = int(np.sum(negatives > best_positive + 1e-8))
    tied = int(np.sum(np.isclose(negatives, best_positive, rtol=1e-6, atol=1e-8)))
    rank = 1.0 + better + 0.5 * tied
    return rank, True, rank == 1.0


@torch.no_grad()
def learned_scores(
    model: ConservativeReranker,
    arrays: dict[str, np.ndarray],
    query_indices: np.ndarray,
    variant: str,
    standardizer: Standardizer,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[float]]:
    model.eval()
    score_map: dict[int, np.ndarray] = {}
    allowed_map: dict[int, np.ndarray] = {}
    gates: list[float] = []
    offsets = arrays["query_offsets"]
    for start in range(0, len(query_indices), batch_size):
        batch_queries = query_indices[start : start + batch_size]
        features, query, _, base, mask, _ = make_batch(
            arrays, batch_queries, variant, standardizer, device
        )
        scores, gate, _ = model(features, query, base)
        scores_np = scores.cpu().numpy()
        mask_np = mask.cpu().numpy()
        gates.extend(gate.cpu().numpy().tolist())
        for row, query_index in enumerate(batch_queries):
            raw_start, raw_end = int(offsets[query_index]), int(offsets[query_index + 1])
            raw = arrays["candidate_features"][raw_start:raw_end]
            allowed = candidate_mask(raw, variant)
            count = int(mask_np[row].sum())
            score_map[int(query_index)] = scores_np[row, :count]
            allowed_map[int(query_index)] = allowed
    return score_map, allowed_map, gates


def evaluate_method(
    arrays: dict[str, np.ndarray],
    query_indices: np.ndarray,
    method: str,
    num_entities: int,
    learned: tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[float]] | None = None,
) -> tuple[dict, np.ndarray]:
    offsets = arrays["query_offsets"]
    ranks = np.empty(len(query_indices), dtype=np.float64)
    covered = 0
    for position, query in enumerate(query_indices):
        start, end = int(offsets[query]), int(offsets[query + 1])
        raw = arrays["candidate_features"][start:end]
        labels = arrays["label"][start:end]
        if learned is None:
            scores, allowed = fixed_scores(raw, method)
        else:
            score_map, allowed_map, _ = learned
            scores, allowed = score_map[int(query)], allowed_map[int(query)]
            labels = labels[allowed]
            allowed = np.ones(len(scores), dtype=np.bool_)
        rank, present, _ = ranks_from_scores(scores, labels, allowed, num_entities)
        ranks[position] = rank
        covered += present
    rr = 1.0 / ranks
    metrics = {
        "queries": int(len(query_indices)),
        "mrr": float(rr.mean()),
        "hits1": float(np.mean(ranks <= 1.0)),
        "hits3": float(np.mean(ranks <= 3.0)),
        "hits10": float(np.mean(ranks <= 10.0)),
        "answer_coverage": covered / max(1, len(query_indices)),
        "covered_mrr": float(rr[ranks < num_entities].mean()) if covered else 0.0,
    }
    if learned is not None:
        metrics["mean_gate"] = float(np.mean(learned[2])) if learned[2] else 0.0
    return metrics, rr


def preservation_metrics(base_rr: np.ndarray, method_rr: np.ndarray) -> dict:
    base_correct = np.isclose(base_rr, 1.0)
    method_correct = np.isclose(method_rr, 1.0)
    denominator = int(base_correct.sum())
    return {
        "base_correct_top1": denominator,
        "preserved_correct_top1": int((base_correct & method_correct).sum()),
        "preservation_rate": float((base_correct & method_correct).sum() / max(1, denominator)),
        "harmed_correct_top1": int((base_correct & ~method_correct).sum()),
        "rescued_to_top1": int((~base_correct & method_correct).sum()),
    }


def moving_block_ci(
    diff: np.ndarray,
    times: np.ndarray,
    width: int,
    draws: int,
    seed: int,
) -> list[float]:
    days = np.unique(times)
    sums = np.asarray([diff[times == day].sum() for day in days])
    counts = np.asarray([(times == day).sum() for day in days])
    block = min(max(1, int(width)), len(days))
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0, len(days), size=(draws, int(math.ceil(len(days) / block)))
    )
    indices = ((starts[:, :, None] + np.arange(block)) % len(days)).reshape(draws, -1)
    indices = indices[:, : len(days)]
    estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def compare_to_base(
    base_rr: np.ndarray,
    method_rr: np.ndarray,
    times: np.ndarray,
    relations: np.ndarray,
    base_relations: int,
    width: int,
    draws: int,
    seed: int,
) -> dict:
    diff = method_rr - base_rr
    forward = relations < base_relations
    return {
        "delta_mrr": float(diff.mean()),
        "ci95_moving_block": moving_block_ci(diff, times, width, draws, seed),
        "direction_delta": [
            float(diff[forward].mean()) if forward.any() else None,
            float(diff[~forward].mean()) if (~forward).any() else None,
        ],
        "positive_timestamps": int(
            sum(diff[times == timestamp].mean() > 0 for timestamp in np.unique(times))
        ),
        "timestamps": int(len(np.unique(times))),
        **preservation_metrics(base_rr, method_rr),
    }


@torch.no_grad()
def selection_mrr(
    model: ConservativeReranker,
    arrays: dict[str, np.ndarray],
    query_indices: np.ndarray,
    variant: str,
    standardizer: Standardizer,
    device: torch.device,
    batch_size: int,
    num_entities: int,
) -> float:
    predictions = learned_scores(
        model, arrays, query_indices, variant, standardizer, device, batch_size
    )
    metrics, _ = evaluate_method(
        arrays, query_indices, variant, num_entities, predictions
    )
    return float(metrics["mrr"])


def train_variant(
    arrays: dict[str, np.ndarray],
    train_queries: np.ndarray,
    selection_queries: np.ndarray,
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
    num_entities: int,
) -> tuple[ConservativeReranker, Standardizer, list[dict], int]:
    standardizer = fit_standardizer(arrays, train_queries, variant)
    model = ConservativeReranker(
        len(CANDIDATE_FEATURES),
        len(QUERY_FEATURES),
        args.hidden_size,
        args.max_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)
    best_mrr = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = generator.permutation(train_queries)
        total_loss = 0.0
        total_usable = 0
        total_preserved = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_queries = shuffled[start : start + args.batch_size]
            candidate, query, labels, base, mask, source_base = make_batch(
                arrays, batch_queries, variant, standardizer, device
            )
            optimizer.zero_grad(set_to_none=True)
            scores, gate, _ = model(candidate, query, base)
            rank_loss, usable = pairwise_ranking_loss(
                scores, labels, mask, args.margin, args.hard_negatives
            )
            if usable == 0:
                continue
            keep_loss, protected = preservation_loss(
                scores,
                base,
                labels,
                mask,
                source_base,
                args.distillation_temperature,
            )
            loss = (
                rank_loss
                + args.preservation_weight * keep_loss
                + args.gate_weight * gate.mean()
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * usable
            total_usable += usable
            total_preserved += protected
        current_mrr = selection_mrr(
            model,
            arrays,
            selection_queries,
            variant,
            standardizer,
            device,
            args.batch_size,
            num_entities,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_usable),
            "usable_train_queries": total_usable,
            "protected_train_queries": total_preserved,
            "selection_mrr": current_mrr,
        }
        history.append(row)
        print(variant, json.dumps(row), flush=True)
        if current_mrr > best_mrr + 1e-8:
            best_mrr = current_mrr
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{variant} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, standardizer, history, best_epoch


def state_payload(
    model: ConservativeReranker,
    standardizer: Standardizer,
    variant: str,
    metadata: dict,
    args: argparse.Namespace,
) -> dict:
    return {
        "version": 1,
        "variant": variant,
        "model_state": model.cpu().state_dict(),
        "candidate_mean": standardizer.candidate_mean,
        "candidate_scale": standardizer.candidate_scale,
        "query_mean": standardizer.query_mean,
        "query_scale": standardizer.query_scale,
        "candidate_features": list(CANDIDATE_FEATURES),
        "query_features": list(QUERY_FEATURES),
        "hidden_size": args.hidden_size,
        "max_residual": args.max_residual,
        "examples_metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)

    examples_path = resolve_path(args.examples)
    report_path = resolve_path(args.output)
    metadata_path = examples_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing examples metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance_verified = metadata.get("rules_provenance_verified") is True
    if not provenance_verified and not args.allow_unverified_examples:
        raise ValueError(
            "Examples use rules without verified strict-prefix provenance. "
            "Rebuild with prefix-mined rules; the override is for smoke tests only."
        )
    with np.load(examples_path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    validate_arrays(arrays, metadata)
    data_dir = Path(metadata["data_dir"])
    # entity2id contains schema only; no validation/test facts are accessed.
    num_entities = len(json.loads((data_dir / "entity2id.json").read_text(encoding="utf-8")))
    train_queries, selection_queries, confirmation_queries, cutoffs = chronological_split(
        arrays["query_time"], args.selection_fraction, args.confirmation_fraction
    )

    base_relations = int(metadata["total_base_relations"])
    split_inventory = {
        "train_queries": int(len(train_queries)),
        "selection_queries": int(len(selection_queries)),
        "confirmation_queries": int(len(confirmation_queries)),
        **cutoffs,
    }
    confirmation_times = arrays["query_time"][confirmation_queries]
    confirmation_relations = arrays["query_relation"][confirmation_queries]
    selection_results = {}
    confirmation_results = {}
    confirmation_rr = {}

    for method in FIXED_METHODS:
        selection_results[method], _ = evaluate_method(
            arrays, selection_queries, method, num_entities
        )
        confirmation_results[method], confirmation_rr[method] = evaluate_method(
            arrays, confirmation_queries, method, num_entities
        )

    checkpoints = {}
    training_history = {}
    best_epochs = {}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        # Reset per treatment so capacity comparisons do not inherit RNG state.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model, standardizer, history, best_epoch = train_variant(
            arrays,
            train_queries,
            selection_queries,
            variant,
            args,
            device,
            num_entities,
        )
        selection_predictions = learned_scores(
            model,
            arrays,
            selection_queries,
            variant,
            standardizer,
            device,
            args.batch_size,
        )
        confirmation_predictions = learned_scores(
            model,
            arrays,
            confirmation_queries,
            variant,
            standardizer,
            device,
            args.batch_size,
        )
        selection_results[variant], _ = evaluate_method(
            arrays, selection_queries, variant, num_entities, selection_predictions
        )
        confirmation_results[variant], confirmation_rr[variant] = evaluate_method(
            arrays, confirmation_queries, variant, num_entities, confirmation_predictions
        )
        checkpoint_path = report_path.with_name(report_path.stem + f"_{variant}.pt")
        torch.save(
            state_payload(model, standardizer, variant, metadata, args), checkpoint_path
        )
        checkpoints[variant] = str(checkpoint_path)
        training_history[variant] = history
        best_epochs[variant] = best_epoch
        model.to(device)

    comparisons = {}
    base_rr = confirmation_rr["tlogic"]
    for method, values in confirmation_rr.items():
        if method == "tlogic":
            continue
        comparisons[f"{method}_vs_tlogic"] = compare_to_base(
            base_rr,
            values,
            confirmation_times,
            confirmation_relations,
            base_relations,
            args.bootstrap_width,
            args.bootstrap_draws,
            args.seed,
        )
    planned_direct_contrasts = (
        ("fixed_official", "fixed_all"),
        ("fixed_official", "fixed_random"),
        ("learned_official", "learned_all"),
        ("learned_official", "learned_random"),
    )
    for treatment, control in planned_direct_contrasts:
        if treatment not in confirmation_rr or control not in confirmation_rr:
            continue
        comparisons[f"{treatment}_vs_{control}"] = compare_to_base(
            confirmation_rr[control],
            confirmation_rr[treatment],
            confirmation_times,
            confirmation_relations,
            base_relations,
            args.bootstrap_width,
            args.bootstrap_draws,
            args.seed,
        )

    gate_conditions = {}
    if all(name in confirmation_rr for name in ("learned_official", "learned_all", "learned_random")):
        official = comparisons["learned_official_vs_tlogic"]
        gate_conditions = {
            "delta_at_least_0_001": official["delta_mrr"] >= 0.001,
            "block_ci_lower_positive": official["ci95_moving_block"][0] > 0.0,
            "both_directions_positive": all(value is not None and value > 0 for value in official["direction_delta"]),
            "beats_learned_all": confirmation_results["learned_official"]["mrr"] > confirmation_results["learned_all"]["mrr"],
            "beats_learned_random": confirmation_results["learned_official"]["mrr"] > confirmation_results["learned_random"]["mrr"],
            "preserves_99_percent_base_top1": official["preservation_rate"] >= 0.99,
        }

    report = {
        "version": 1,
        "status": (
            "exploratory_train_only_internal_confirmation"
            if provenance_verified
            else "smoke_only_unverified_rule_provenance"
        ),
        "examples": str(examples_path),
        "examples_metadata": metadata,
        "seed": args.seed,
        "device": str(device),
        "num_entities": num_entities,
        "split": split_inventory,
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"examples", "output", "variants"}
        },
        "variants": args.variants,
        "selection_results": selection_results,
        "confirmation_results": confirmation_results,
        "paired_confirmation_comparisons": comparisons,
        "best_epochs": best_epochs,
        "training_history": training_history,
        "checkpoints": checkpoints,
        "progression_gate_conditions": gate_conditions,
        "progression_gate_passed": (
            provenance_verified and bool(gate_conditions) and all(gate_conditions.values())
        ),
        "rules_provenance_verified": provenance_verified,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "model_sha256": sha256((SCRIPT_DIR / "model.py").resolve()),
        "elapsed_seconds": time.time() - started,
        "validation_read": False,
        "test_read": False,
    }
    json_dump(report, report_path)
    print(json.dumps({
        "report": str(report_path),
        "confirmation_results": confirmation_results,
        "progression_gate_conditions": gate_conditions,
        "progression_gate_passed": report["progression_gate_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
