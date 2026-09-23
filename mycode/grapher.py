import json
from pathlib import Path

import numpy as np


class Grapher:
    """Load a temporal knowledge graph and build ID-based index arrays."""

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)

        self.entity2id = self._load_json("entity2id.json")
        self.relation2id_original = self._load_json("relation2id.json")
        self.ts2id = self._load_json("ts2id.json")

        self.relation2id = dict(self.relation2id_original)
        inverse_start = len(self.relation2id_original)
        for relation_id, relation_name in enumerate(self.relation2id_original):
            self.relation2id[f"_{relation_name}"] = inverse_start + relation_id

        self.id2entity = {value: key for key, value in self.entity2id.items()}
        self.id2relation = {value: key for key, value in self.relation2id.items()}
        self.id2ts = {value: key for key, value in self.ts2id.items()}

        num_relations = len(self.relation2id_original)
        self.inv_relation_id = {}
        for relation_id in range(num_relations):
            self.inv_relation_id[relation_id] = relation_id + num_relations
        for relation_id in range(num_relations, 2 * num_relations):
            self.inv_relation_id[relation_id] = relation_id % num_relations

        self.train_idx = self._build_store("train.txt")
        self.valid_idx = self._build_store("valid.txt")
        self.test_idx = self._build_store("test.txt")
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

    def _load_json(self, filename):
        with (self.dataset_dir / filename).open(encoding="utf-8") as file:
            return json.load(file)

    def _read_quads(self, filename):
        with (self.dataset_dir / filename).open(encoding="utf-8") as file:
            return [line.rstrip("\n").split("\t") for line in file]

    def _map_to_ids(self, quads):
        subjects = [self.entity2id[quad[0]] for quad in quads]
        relations = [self.relation2id[quad[1]] for quad in quads]
        objects = [self.entity2id[quad[2]] for quad in quads]
        timestamps = [self.ts2id[quad[3]] for quad in quads]
        return np.column_stack((subjects, relations, objects, timestamps))

    def _add_inverse_quads(self, quads):
        inverse_relations = [self.inv_relation_id[int(rel)] for rel in quads[:, 1]]
        inverse_quads = np.column_stack(
            (
                quads[:, 2],
                inverse_relations,
                quads[:, 0],
                quads[:, 3],
            )
        )
        return np.vstack((quads, inverse_quads))

    def _build_store(self, filename):
        quads = self._read_quads(filename)
        indexed_quads = self._map_to_ids(quads)
        return self._add_inverse_quads(indexed_quads)
