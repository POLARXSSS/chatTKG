import numpy as np


def store_neighbors(quads):
    """Group outgoing edges by source entity."""
    return {
        node: quads[quads[:, 0] == node]
        for node in set(quads[:, 0])
    }


def store_edges(quads):
    """Group edges by relation."""
    return {
        relation: quads[quads[:, 1] == relation]
        for relation in set(quads[:, 1])
    }


class Temporal_Walk:
    """Sample cyclic temporal random walks used for rule mining."""

    def __init__(self, learn_data, inv_relation_id, transition_distr):
        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_distr = transition_distr
        self.neighbors = store_neighbors(learn_data)
        self.edges = store_edges(learn_data)

    def sample_start_edge(self, relation_id):
        relation_edges = self.edges[relation_id]
        return relation_edges[np.random.choice(len(relation_edges))]

    def sample_next_edge(self, filtered_edges, current_timestamp):
        if self.transition_distr == "unif":
            return filtered_edges[np.random.choice(len(filtered_edges))]

        probabilities = np.exp(filtered_edges[:, 3] - current_timestamp)
        try:
            probabilities = probabilities / probabilities.sum()
            return filtered_edges[
                np.random.choice(len(filtered_edges), p=probabilities)
            ]
        except ValueError:
            # The exponential distribution can underflow for very old timestamps.
            return filtered_edges[np.random.choice(len(filtered_edges))]

    def transition_step(
        self, current_node, current_timestamp, previous_edge, start_node, step, walk_length
    ):
        candidates = self.neighbors[current_node]

        if step == 1:
            candidates = candidates[candidates[:, 3] < current_timestamp]
        else:
            candidates = candidates[candidates[:, 3] <= current_timestamp]
            inverse_edge = np.array(
                [
                    current_node,
                    self.inv_relation_id[previous_edge[1]],
                    previous_edge[0],
                    current_timestamp,
                ]
            )
            inverse_rows = np.where(np.all(candidates == inverse_edge, axis=1))
            candidates = np.delete(candidates, inverse_rows, axis=0)

        if step == walk_length - 1:
            candidates = candidates[candidates[:, 2] == start_node]

        if len(candidates):
            return self.sample_next_edge(candidates, current_timestamp)
        return np.array([])

    def sample_walk(self, walk_length, relation_id):
        walk = {}
        previous_edge = self.sample_start_edge(relation_id)
        start_node = previous_edge[0]
        current_node = previous_edge[2]
        current_timestamp = previous_edge[3]

        walk["entities"] = [start_node, current_node]
        walk["relations"] = [previous_edge[1]]
        walk["timestamps"] = [current_timestamp]

        for step in range(1, walk_length):
            next_edge = self.transition_step(
                current_node,
                current_timestamp,
                previous_edge,
                start_node,
                step,
                walk_length,
            )
            if not len(next_edge):
                return False, walk

            current_node = next_edge[2]
            current_timestamp = next_edge[3]
            walk["relations"].append(next_edge[1])
            walk["entities"].append(current_node)
            walk["timestamps"].append(current_timestamp)
            previous_edge = next_edge

        return True, walk
