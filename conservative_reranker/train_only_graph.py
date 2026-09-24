from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TrainOnlyGrapher:
    """Load graph schemas and train.txt without opening validation/test files."""

    def __init__(self, dataset_dir: str | Path) -> None:
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

        relation_count = len(self.relation2id_original)
        self.inv_relation_id = {
            relation_id: (
                relation_id + relation_count
                if relation_id < relation_count
                else relation_id - relation_count
            )
            for relation_id in range(2 * relation_count)
        }
        self.train_idx = self._build_train_store()

    def _load_json(self, filename: str) -> dict:
        with (self.dataset_dir / filename).open(encoding="utf-8") as handle:
            return json.load(handle)

    def _build_train_store(self) -> np.ndarray:
        with (self.dataset_dir / "train.txt").open(encoding="utf-8") as handle:
            quads = [line.rstrip("\n").split("\t") for line in handle]
        indexed = np.asarray(
            [
                (
                    self.entity2id[subject],
                    self.relation2id[relation],
                    self.entity2id[object_],
                    self.ts2id[timestamp],
                )
                for subject, relation, object_, timestamp in quads
            ],
            dtype=np.int64,
        ).reshape(-1, 4)
        inverse = np.column_stack(
            (
                indexed[:, 2],
                np.asarray(
                    [self.inv_relation_id[int(value)] for value in indexed[:, 1]],
                    dtype=np.int64,
                ),
                indexed[:, 0],
                indexed[:, 3],
            )
        )
        return np.vstack((indexed, inverse)).astype(np.int64, copy=False)
