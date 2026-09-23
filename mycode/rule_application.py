import json
import os

import numpy as np
import pandas as pd

from temporal_walk import store_edges


def admission_confidence(rule):
    """Return the raw TLogic confidence used for rule admission.

    Fused rule files keep ``conf`` for the selected inference score, so it
    must not also control the statistical pruning threshold.  Older files do
    not have ``tlogic_conf`` and are reconstructed from their supports.
    """
    tlogic_conf = rule.get("tlogic_conf")
    if tlogic_conf is not None:
        return float(tlogic_conf)

    rule_support = rule.get("rule_supp")
    body_support = rule.get("body_supp")
    if rule_support is not None and body_support:
        return float(rule_support) / float(body_support)

    return float(rule.get("conf", 0.0))


def filter_rules(rules_dict, min_conf, min_body_supp, rule_lengths):
    """Keep rules that pass raw TLogic, support and length filters."""
    filtered = {}
    for relation_id, rules in rules_dict.items():
        filtered[relation_id] = [
            rule
            for rule in rules
            if admission_confidence(rule) >= min_conf
            and rule["body_supp"] >= min_body_supp
            and len(rule["body_rels"]) in rule_lengths
        ]
    return filtered


def get_window_edges(all_data, query_timestamp, learn_edges, window=-1):
    """Return edges used for applying rules at a query timestamp."""
    if window > 0:
        mask = (all_data[:, 3] < query_timestamp) & (
            all_data[:, 3] >= query_timestamp - window
        )
        return store_edges(all_data[mask])
    if window == 0:
        return store_edges(all_data[all_data[:, 3] < query_timestamp])
    return learn_edges


def match_body_relations(rule, edges, query_subject):
    """Collect candidate edges for each body relation from the query subject."""
    body_relations = rule["body_rels"]
    try:
        relation_edges = edges[body_relations[0]]
        first_edges = relation_edges[relation_edges[:, 0] == query_subject]
        walk_edges = [np.hstack((first_edges[:, 0:1], first_edges[:, 2:4]))]
        current_targets = np.array(list(set(walk_edges[0][:, 1])))

        for relation_id in body_relations[1:]:
            try:
                relation_edges = edges[relation_id]
                mask = np.any(relation_edges[:, 0] == current_targets[:, None], axis=0)
                matched_edges = relation_edges[mask]
                walk_edges.append(
                    np.hstack((matched_edges[:, 0:1], matched_edges[:, 2:4]))
                )
                current_targets = np.array(list(set(walk_edges[-1][:, 1])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        return [[]]

    return walk_edges


def get_walks(rule, walk_edges):
    """Build a DataFrame of time-ordered walks matching the rule body."""
    edge_frames = []
    first_frame = pd.DataFrame(
        walk_edges[0],
        columns=["entity_0", "entity_1", "timestamp_0"],
        dtype=np.uint16,
    )
    if not rule["var_constraints"]:
        del first_frame["entity_0"]
    edge_frames.append(first_frame)

    for step, edges in enumerate(walk_edges[1:], start=1):
        frame = pd.DataFrame(
            edges,
            columns=[
                f"entity_{step}",
                f"entity_{step + 1}",
                f"timestamp_{step}",
            ],
            dtype=np.uint16,
        )
        edge_frames.append(frame)

    rule_walks = edge_frames[0]
    edge_frames[0] = edge_frames[0][0:0]
    for step in range(1, len(edge_frames)):
        rule_walks = pd.merge(
            rule_walks, edge_frames[step], on=f"entity_{step}"
        )
        rule_walks = rule_walks[
            rule_walks[f"timestamp_{step - 1}"]
            <= rule_walks[f"timestamp_{step}"]
        ]
        if not rule["var_constraints"]:
            del rule_walks[f"entity_{step}"]
        edge_frames[step] = edge_frames[step][0:0]

    for step in range(1, len(rule["body_rels"])):
        del rule_walks[f"timestamp_{step}"]

    return rule_walks


def check_var_constraints(var_constraints, rule_walks):
    """Keep walks whose repeated variables refer to the same entity."""
    for constraint in var_constraints:
        for index in range(len(constraint) - 1):
            rule_walks = rule_walks[
                rule_walks[f"entity_{constraint[index]}"]
                == rule_walks[f"entity_{constraint[index + 1]}"]
            ]
    return rule_walks


def _join_walk_edges(entities, timestamps, next_edges):
    """Join current walks with the next body relation using NumPy."""
    if len(entities) == 0 or len(next_edges) == 0:
        return entities[:0], timestamps[:0]

    left_keys = entities[:, -1]
    right_subjects = next_edges[:, 0]
    order = np.argsort(right_subjects, kind="mergesort")
    sorted_subjects = right_subjects[order]
    starts = np.searchsorted(sorted_subjects, left_keys, side="left")
    ends = np.searchsorted(sorted_subjects, left_keys, side="right")
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return entities[:0], timestamps[:0]

    left_indices = np.repeat(np.arange(len(entities)), counts)
    starts_repeated = np.repeat(starts, counts)
    group_starts = np.repeat(np.cumsum(counts) - counts, counts)
    within_group = np.arange(total) - group_starts
    right_indices = order[starts_repeated + within_group]

    time_mask = (
        timestamps[left_indices, -1] <= next_edges[right_indices, 2]
    )
    left_indices = left_indices[time_mask]
    right_indices = right_indices[time_mask]
    if len(left_indices) == 0:
        return entities[:0], timestamps[:0]

    joined_entities = np.concatenate(
        (entities[left_indices], next_edges[right_indices, 1:2]), axis=1
    )
    joined_timestamps = np.concatenate(
        (timestamps[left_indices], next_edges[right_indices, 2:3]), axis=1
    )
    return joined_entities, joined_timestamps


def get_walks_arrays(rule, walk_edges):
    """Fast NumPy version of get_walks for rule application."""
    num_entities = len(rule["body_rels"]) + 1
    num_timestamps = len(rule["body_rels"])
    if not walk_edges or len(walk_edges[0]) == 0:
        return (
            np.empty((0, num_entities), dtype=np.uint16),
            np.empty((0, num_timestamps), dtype=np.uint16),
        )

    first_edges = walk_edges[0]
    entities = first_edges[:, :2]
    timestamps = first_edges[:, 2:3]

    for next_edges in walk_edges[1:]:
        entities, timestamps = _join_walk_edges(
            entities, timestamps, next_edges
        )
        if len(entities) == 0:
            break

    if rule["var_constraints"] and len(entities):
        keep = np.ones(len(entities), dtype=bool)
        for constraint in rule["var_constraints"]:
            for left, right in zip(constraint, constraint[1:]):
                keep &= entities[:, left] == entities[:, right]
        entities = entities[keep]
        timestamps = timestamps[keep]

    return entities, timestamps


def get_candidate_scores_arrays(
    rule,
    entities,
    timestamps,
    query_timestamp,
    candidates,
    score_configs,
    active_indices,
):
    """Fast NumPy version of get_candidates for rule application."""
    if len(entities) == 0:
        return candidates

    candidate_ids = entities[:, -1]
    first_timestamps = timestamps[:, 0].astype(np.int64)
    unique_candidates, inverse = np.unique(
        candidate_ids, return_inverse=True
    )
    max_timestamps = np.full(len(unique_candidates), -1, dtype=np.int64)
    np.maximum.at(max_timestamps, inverse, first_timestamps)

    score_conf = rule.get("score_conf")
    if score_conf is not None:
        rule_score = float(score_conf)
    elif "rule_supp" in rule and "body_supp" in rule:
        rule_score = rule["rule_supp"] / (rule["body_supp"] + 0)
    else:
        rule_score = float(rule["conf"])
    rule_score = float(np.clip(rule_score, 0.0, 1.0))

    for score_index in active_indices:
        decay_rate, weight = score_configs[score_index]
        time_scores = np.exp(
            decay_rate * (max_timestamps - float(query_timestamp))
        )
        scores = weight * rule_score + (1 - weight) * time_scores
        scores = np.clip(scores, 0.0, 1.0)
        for candidate, score in zip(unique_candidates, scores):
            candidates[score_index].setdefault(int(candidate), []).append(
                # The original pandas path converts each rule score to
                # float32 before noisy-or aggregation.
                float(np.float32(score))
            )

    return candidates


def get_candidates(
    rule, rule_walks, query_timestamp, candidates, score_func, score_args, active_indices
):
    """Add candidate tail entities and their rule scores."""
    tail_column = f"entity_{len(rule['body_rels'])}"
    for candidate in set(rule_walks[tail_column]):
        candidate_walks = rule_walks[rule_walks[tail_column] == candidate]
        for score_index in active_indices:
            score = score_func(
                rule,
                candidate_walks,
                query_timestamp,
                *score_args[score_index],
            ).astype(np.float32)
            candidates[score_index].setdefault(candidate, []).append(score)
    return candidates


def save_candidates(
    rules_file,
    output_dir,
    all_candidates,
    rule_lengths,
    window,
    score_name,
    prefix="",
):
    """Write candidate predictions to a JSON file."""
    candidates = {
        int(query_index): {int(entity): value for entity, value in scores.items()}
        for query_index, scores in all_candidates.items()
    }

    base_name = os.path.splitext(rules_file)[0]
    if prefix:
        base_name = f"{prefix}_{base_name}"
    if base_name.endswith("_rules"):
        base_name = base_name[:-6]
    if "final_rules_for_inference" in base_name:
        base_name = "llm_optimized"

    filename = (
        f"{base_name}_cands_r{rule_lengths}_w{window}_{score_name}.json"
    ).replace(" ", "")
    with open(output_dir + filename, "w", encoding="utf-8") as file:
        json.dump(candidates, file)
    return filename
