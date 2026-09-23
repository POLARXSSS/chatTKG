import itertools
import json
import os
from collections import Counter

import numpy as np


class Rule_Learner:
    def __init__(self, edges, id2relation, inv_relation_id, dataset):
        self.edges = edges
        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id
        self.found_rules = []
        self.rules_dict = {}
        self.output_dir = f"../output/{dataset}/"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_rule(self, walk):
        rule = {
            "head_rel": int(walk["relations"][0]),
            "body_rels": [
                self.inv_relation_id[relation]
                for relation in walk["relations"][1:][::-1]
            ],
            "var_constraints": self.define_var_constraints(
                walk["entities"][1:][::-1]
            ),
        }

        if rule in self.found_rules:
            return

        self.found_rules.append(rule.copy())
        rule["conf"], rule["rule_supp"], rule["body_supp"] = self.estimate_confidence(
            rule
        )
        if rule["conf"]:
            self.update_rules_dict(rule)

    def define_var_constraints(self, entities):
        constraints = [
            [index for index, entity in enumerate(entities) if entity == value]
            for value in set(entities)
        ]
        return sorted([item for item in constraints if len(item) > 1])

    def estimate_confidence(self, rule, num_samples=500):
        bodies = []
        for _ in range(num_samples):
            sampled, body = self.sample_body(
                rule["body_rels"], rule["var_constraints"]
            )
            if sampled:
                bodies.append(body)

        bodies.sort()
        unique_bodies = list(item for item, _ in itertools.groupby(bodies))
        body_support = len(unique_bodies)

        confidence = 0
        rule_support = 0
        if body_support:
            rule_support = self.calculate_rule_support(
                unique_bodies, rule["head_rel"]
            )
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support

    def sample_body(self, body_relations, variable_constraints):
        first_edges = self.edges[body_relations[0]]
        current_edge = first_edges[np.random.choice(len(first_edges))]
        current_timestamp = current_edge[3]
        current_node = current_edge[2]
        body = [current_edge[0], current_timestamp, current_node]

        for relation_id in body_relations[1:]:
            candidate_edges = self.edges[relation_id]
            mask = (candidate_edges[:, 0] == current_node) & (
                candidate_edges[:, 3] >= current_timestamp
            )
            filtered_edges = candidate_edges[mask]
            if not len(filtered_edges):
                return False, body

            current_edge = filtered_edges[np.random.choice(len(filtered_edges))]
            current_timestamp = current_edge[3]
            current_node = current_edge[2]
            body.extend([current_timestamp, current_node])

        if variable_constraints:
            body_constraints = self.define_var_constraints(body[::2])
            if body_constraints != variable_constraints:
                return False, body

        return True, body

    def calculate_rule_support(self, unique_bodies, head_relation):
        head_edges = self.edges[head_relation]
        rule_support = 0
        for body in unique_bodies:
            mask = (
                (head_edges[:, 0] == body[0])
                & (head_edges[:, 2] == body[-1])
                & (head_edges[:, 3] > body[-2])
            )
            if np.any(mask):
                rule_support += 1
        return rule_support

    def update_rules_dict(self, rule):
        self.rules_dict.setdefault(rule["head_rel"], []).append(rule)

    def sort_rules_dict(self):
        for relation_id in self.rules_dict:
            self.rules_dict[relation_id] = sorted(
                self.rules_dict[relation_id], key=lambda rule: rule["conf"], reverse=True
            )

    def save_rules(self, timestamp, rule_lengths, num_walks, transition_distr, seed):
        filename = (
            f"{timestamp}_r{rule_lengths}_n{num_walks}_"
            f"{transition_distr}_s{seed}_rules.json"
        ).replace(" ", "")
        rules_dict = {int(key): value for key, value in self.rules_dict.items()}
        with open(self.output_dir + filename, "w", encoding="utf-8") as file:
            json.dump(rules_dict, file)

    def save_rules_verbalized(
        self, timestamp, rule_lengths, num_walks, transition_distr, seed
    ):
        filename = (
            f"{timestamp}_r{rule_lengths}_n{num_walks}_"
            f"{transition_distr}_s{seed}_rules.txt"
        ).replace(" ", "")
        content = "\n".join(
            verbalize_rule(rule, self.id2relation)
            for rules in self.rules_dict.values()
            for rule in rules
        )
        with open(self.output_dir + filename, "w", encoding="utf-8") as file:
            file.write(content + "\n")


def verbalize_rule(rule, id2relation):
    if rule["var_constraints"]:
        constraints = rule["var_constraints"]
        used = [item for group in constraints for item in group]
        for index in range(len(rule["body_rels"]) + 1):
            if index not in used:
                constraints.append([index])
        constraints = sorted(constraints)
    else:
        constraints = [[index] for index in range(len(rule["body_rels"]) + 1)]

    head_object_group_index = next(
        group_index
        for group_index, group in enumerate(constraints)
        if len(rule["body_rels"]) in group
    )
    rule_text = (
        f"{rule['conf']:8.6f}  {rule['rule_supp']:4}  {rule['body_supp']:4}  "
        f"{id2relation[rule['head_rel']]}(X0,X{head_object_group_index},"
        f"T{len(rule['body_rels'])}) <- "
    )

    for index, body_relation in enumerate(rule["body_rels"]):
        subject_group_index = next(
            group_index
            for group_index, group in enumerate(constraints)
            if index in group
        )
        object_group_index = next(
            group_index
            for group_index, group in enumerate(constraints)
            if index + 1 in group
        )
        rule_text += (
            f"{id2relation[body_relation]}(X{subject_group_index},"
            f"X{object_group_index},T{index}), "
        )

    return rule_text[:-2]


def rules_statistics(rules_dict):
    rule_count = sum(len(rules) for rules in rules_dict.values())
    lengths = [
        len(rule["body_rels"])
        for rules in rules_dict.values()
        for rule in rules
    ]
    length_counts = sorted(Counter(lengths).items())
    print("Number of relations with rules: ", len(rules_dict))
    print("Total number of rules: ", rule_count)
    print("Number of rules by length: ", length_counts)
