import abc

import numpy as np

from gng import GrowingNeuralGas


__author__ = 'Siarshai'

class RobustGrowingNeuralGas(GrowingNeuralGas):
    def __init__(self, weights, winner_learning_rate, neighbours_learning_rate, **kwargs):
        super().__init__(weights, winner_learning_rate, neighbours_learning_rate, **kwargs)
        self._X_cache = None

    @abc.abstractmethod
    def _recompute_smoothing_factor(self, neuron_id, distance):
        return None

    @abc.abstractmethod
    def _init_smoothing_factor(self, neuron_id, w, X):
        pass

    def _update_weights_and_connections(self, x):
        raw_vector_difference = self.neurons["value"] - x
        distances_from_x = np.linalg.norm(raw_vector_difference, axis=1)
        closest_neuron_idx, second_closest_neuron_idx = distances_from_x.argsort()[:2]
        smoothing_factor = self._recompute_smoothing_factor(
                self.neurons["id"][closest_neuron_idx], distances_from_x[closest_neuron_idx])
        self.neurons["error"][closest_neuron_idx] += smoothing_factor * distances_from_x[closest_neuron_idx]
        self.neurons["value"][closest_neuron_idx] -= smoothing_factor * self.winner_learning_rate * raw_vector_difference[closest_neuron_idx]
        idxs = self._get_neighbours_idxs_of_neuron(closest_neuron_idx)
        for idx in idxs:
            smoothing_factor = self._recompute_smoothing_factor(
                    self.neurons["id"][idx], distances_from_x[idx])
            self.neurons["value"][idx] -= smoothing_factor * self.neighbours_learning_rate * raw_vector_difference[idx]
        all_neuron_ids = list(self.neurons["id"])
        closest_neuron_name, second_closest_neuron_name = \
            all_neuron_ids[closest_neuron_idx], all_neuron_ids[second_closest_neuron_idx]
        self.age_of_connections[frozenset([closest_neuron_name, second_closest_neuron_name])] = 0
        return closest_neuron_idx, second_closest_neuron_idx

    def fit(self, X, number_of_epochs):
        self._X_cache = X
        super().fit(X, number_of_epochs)

    def _insert_neuron(self):
        super()._insert_neuron()
        self._init_smoothing_factor(self.neurons["id"][-1], self.neurons["value"][-1], self._X_cache)


class QinRobustGrowingNeuralGas(RobustGrowingNeuralGas):
    def __init__(self, weights, winner_learning_rate, neighbours_learning_rate, **kwargs):
        super().__init__(weights, winner_learning_rate, neighbours_learning_rate, **kwargs)
        self.accumulated_harmonic_distances = {}

    def _recompute_smoothing_factor(self, neuron_id, distance):
        ahd = self.accumulated_harmonic_distances[neuron_id]
        if distance >= ahd:
            self.accumulated_harmonic_distances[neuron_id] = 1/(0.5*(1/ahd + 1/(0.001 + distance)))
            return self.accumulated_harmonic_distances[neuron_id]/distance
        else:
            self.accumulated_harmonic_distances[neuron_id] = 0.5*(ahd + distance)
            return 1.0

    def _pre_epoch(self, epoch):
        super()._pre_epoch(epoch)
        for neuron_id, w in zip(self.neurons["id"], self.neurons["value"]):
            self._init_smoothing_factor(neuron_id, w, self._X_cache)

    def _init_smoothing_factor(self, neuron_id, w, X):
        self.accumulated_harmonic_distances[neuron_id] = sum([1.0/(0.001 + np.linalg.norm(w - x)) for x in X])/len(X)

    def _remove_invalid_neurons(self, closest_neuron_idx):
        invalid_neurons_ids = super()._remove_invalid_neurons(closest_neuron_idx)
        for neuron_id in invalid_neurons_ids:
            del self.accumulated_harmonic_distances[neuron_id]

class SimpleRobustGrowingNeuralGas(RobustGrowingNeuralGas):
    def __init__(self, weights, winner_learning_rate, neighbours_learning_rate,
                 keep_accumulated_distance_factor=0.95, **kwargs):
        super().__init__(weights, winner_learning_rate, neighbours_learning_rate, **kwargs)
        self.keep_accumulated_distance_factor = keep_accumulated_distance_factor
        self.accumulated_distances = {}

    def _recompute_smoothing_factor(self, neuron_id, distance):
        self.accumulated_distances[neuron_id] = (1 - self.keep_accumulated_distance_factor)*distance + \
                self.keep_accumulated_distance_factor*self.accumulated_distances[neuron_id]
        if distance < self.accumulated_distances[neuron_id]:
            return 1.0
        else:
            return self.accumulated_distances[neuron_id]/distance

    def _init_smoothing_factor(self, neuron_id, w, X):
        self.accumulated_distances[neuron_id] = 0.0

    def _remove_invalid_neurons(self, closest_neuron_idx):
        invalid_neurons_ids = super()._remove_invalid_neurons(closest_neuron_idx)
        for neuron_id in invalid_neurons_ids:
            del self.accumulated_distances[neuron_id]

    def fit(self, X, number_of_epochs):
        for neuron_id, w in zip(self.neurons["id"], self.neurons["value"]):
            self._init_smoothing_factor(neuron_id, w, X)
        super().fit(X, number_of_epochs)


