from enum import Enum

import numpy as np


class HistoryLevel(Enum):
    NONE = 0
    EPOCHS = 1
    STEPS = 2

class GrowingNeuralGas(object):
    def __init__(self, weights, winner_learning_rate, neighbours_learning_rate, learning_rate_decay=1.0,
                 edge_max_age=100, populate_iterations_divisor=25, max_neurons=10,
                 insertion_error_decay=0.8, iteration_error_decay=0.99, with_history=HistoryLevel.NONE,
                 force_dying=False, **kwargs):

        # Not really necessary, but for the sake of simplicity (no need to generate initial connections)
        # let's assume there are only 2 neurons
        if len(weights) != 2:
            raise AttributeError("GNG should initially have 2 neurons")
        self.neuron_idx = -1
        neurons = []
        for w in weights:
            self.neuron_idx += 1
            neurons.append((str(self.neuron_idx), w, 0))
        self.neurons = np.array(
                neurons,
                dtype=[("id", '|S10'), ("value", '>f8', weights[0].shape), ("error", '>f8')]
            )
        self.age_of_connections = {
            frozenset([b"0", b"1"]): 0
        }

        self.winner_learning_rate = winner_learning_rate
        self.neighbours_learning_rate = neighbours_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.edge_max_age = edge_max_age
        self.populate_iterations_divisor = populate_iterations_divisor
        self.max_neurons = max_neurons

        self.insertion_error_decay = insertion_error_decay
        self.iteration_error_decay = iteration_error_decay

        self.with_history = with_history

        self.total_epoch = 0
        self.total_sampled = 0
        self.order = []
        self.iteration_num = 0
        self._history = []
        self.force_dying = force_dying

    def _get_neighbours_idxs_of_neuron(self, neuron_idx):
        neuron_id = self.neurons["id"][neuron_idx]
        neighbours_ids = set()
        for keyset in self.age_of_connections:
            if neuron_id in keyset:
                neighbours_ids = neighbours_ids.union(keyset)
        # for the sake of algorithm functionality node is NOT a neighbour to itself
        neighbours_ids.remove(neuron_id)
        full_ids_list = list(self.neurons["id"])
        return [full_ids_list.index(id) for id in neighbours_ids]

    def get_connections_idx_pairs(self):
        ids = list(self.neurons["id"])
        idx_pairs = []
        for keyset in self.age_of_connections.keys():
            pair = []
            for x in keyset:
                pair.append(ids.index(x))
            idx_pairs.append(pair)
        return idx_pairs

    def _pre_fit_one(self, **kwargs):
        self.iteration_num += 1
        if self.with_history == HistoryLevel.STEPS:
            self._history.append({
                "W" : np.array(self.neurons["value"]),
                "connections" : self.get_connections_idx_pairs(),
                "x" : kwargs["x"]
            })

    def _update_weights_and_connections(self, x):
        raw_vector_difference = self.neurons["value"] - x
        distances_from_x = np.linalg.norm(raw_vector_difference, axis=1)
        closest_neuron_idx, second_closest_neuron_idx = distances_from_x.argsort()[:2]
        self.neurons["error"][closest_neuron_idx] += distances_from_x[closest_neuron_idx]
        self.neurons["value"][closest_neuron_idx] -= self.winner_learning_rate * raw_vector_difference[closest_neuron_idx]
        idxs = self._get_neighbours_idxs_of_neuron(closest_neuron_idx)
        for idx in idxs:
            self.neurons["value"][idx] -= self.neighbours_learning_rate * raw_vector_difference[idx]
        all_neuron_ids = list(self.neurons["id"])
        closest_neuron_name, second_closest_neuron_name = \
            all_neuron_ids[closest_neuron_idx], all_neuron_ids[second_closest_neuron_idx]
        self.age_of_connections[frozenset([closest_neuron_name, second_closest_neuron_name])] = 0
        return closest_neuron_idx, second_closest_neuron_idx

    def _remove_invalid_neurons(self, source_neuron_idx):
        all_neuron_ids = list(self.neurons["id"])
        source_neuron_id = all_neuron_ids[source_neuron_idx]
        valid_neurons = set()
        for keyset in list(self.age_of_connections.keys()):
            if self.force_dying or source_neuron_id in keyset:
                self.age_of_connections[keyset] += 1
                if self.age_of_connections[keyset] >= self.edge_max_age:
                    del self.age_of_connections[keyset]
                else:
                    valid_neurons.update(keyset)
            else:
                valid_neurons.update(keyset)
        if len(valid_neurons) != len(all_neuron_ids):
            invalid_neurons_ids = set(all_neuron_ids).difference(valid_neurons)
            invalid_neurons_idxs = [all_neuron_ids.index(id) for id in invalid_neurons_ids]
            valid_indexes = [i for i in range(len(all_neuron_ids)) if i not in invalid_neurons_idxs]
            self.neurons = self.neurons[valid_indexes]
            return invalid_neurons_ids
        return []

    def _should_insert_neuron_predicate(self, **kwargs):
        return len(self.neurons) < self.max_neurons and not self.iteration_num % self.populate_iterations_divisor

    def _insert_neuron(self):
        self.neuron_idx += 1
        largest_error_neuron_idx = np.argmax(self.neurons["error"])
        idxs = self._get_neighbours_idxs_of_neuron(largest_error_neuron_idx)
        max_error, largest_error_neighbour_neuron_idx = 0.0, 0
        for idx in idxs:
            if self.neurons["error"][idx] > max_error:
                largest_error_neighbour_neuron_idx = idx
                max_error = self.neurons["error"][idx]
        new_neuron_weights = (self.neurons["value"][largest_error_neighbour_neuron_idx] +
                2 * self.neurons["value"][largest_error_neuron_idx]) / 3
        new_id = str(self.neuron_idx).encode('UTF-8')
        self.neurons["error"][largest_error_neuron_idx] *= self.insertion_error_decay
        self.neurons["error"][largest_error_neighbour_neuron_idx] *= self.insertion_error_decay
        self.neurons = np.append(self.neurons,
                np.array([(
                           new_id,
                           new_neuron_weights,
                           (self.neurons["error"][largest_error_neighbour_neuron_idx] +
                            self.neurons["error"][largest_error_neuron_idx]) / 2.0
                       )], dtype=self.neurons[0].dtype)
                )
        self.age_of_connections[frozenset([self.neurons["id"][largest_error_neuron_idx], new_id])] = 0
        self.age_of_connections[frozenset([self.neurons["id"][largest_error_neighbour_neuron_idx], new_id])] = 0
        removed_arc = frozenset(
            [str(largest_error_neuron_idx).encode('UTF-8'), str(largest_error_neighbour_neuron_idx).encode('UTF-8')])
        if removed_arc in self.age_of_connections:
            del self.age_of_connections[removed_arc]

    def _post_fit_one(self):
        self.neurons["error"] *= self.iteration_error_decay

    def _post_epoch(self, epoch):
        self.winner_learning_rate *= self.learning_rate_decay
        self.neighbours_learning_rate *= self.learning_rate_decay
        if self.with_history == HistoryLevel.EPOCHS:
            self._history.append({
                "W" : self.neurons["value"],
                "connections" : self.get_connections_idx_pairs()
            })
        print("Neural gas: Ending epoch {} ({} total)".format(epoch + 1, self.total_epoch))

    def _pre_epoch(self, epoch):
        self.total_epoch += 1
        print("Neural gas: Beginning epoch {} ({} total)".format(epoch + 1, self.total_epoch))
        np.random.shuffle(self.order)

    def fit_one(self, x):
        self._pre_fit_one(x=x)
        closest_neuron_idx, second_closest_neuron_idx = self._update_weights_and_connections(x)
        self._remove_invalid_neurons(closest_neuron_idx)
        if self._should_insert_neuron_predicate():
            self._insert_neuron()
        self._post_fit_one()

    def fit(self, X, number_of_epochs):
        self.order = list(range(len(X)))
        for epoch in range(number_of_epochs):
            self._pre_epoch(epoch)
            for i in self.order:
                self.fit_one(X[i])
            self._post_epoch(epoch)

    def fit_p(self, fn, times_to_sample=1500, number_of_epochs_per_batch=1, size_of_batch=8):
        for t in range(times_to_sample):
            self.total_sampled += 1
            print("Sampling {} ({} total)".format(t + 1, self.total_sampled))
            X = fn(t, size_of_batch)
            self.fit(X, number_of_epochs_per_batch)
            if self.with_history == HistoryLevel.EPOCHS:
                self._history[-1]["x"] = X

    def get_weights(self):
        return self.neurons["value"]

    def get_weights_history(self):
        return [record["W"] for record in self._history]

    def get_connections_history(self):
        return [record["connections"] for record in self._history]

    def get_samples_history(self):
        if len(self._history) and "x" in self._history[0]:
            return [record["x"] for record in self._history]
        return []

