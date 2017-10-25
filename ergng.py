from rgng import SimpleRobustGrowingNeuralGas
import numpy as np


class ExpandingSRGNG(SimpleRobustGrowingNeuralGas):
    def __init__(self, weights, winner_learning_rate, neighbours_learning_rate,
                 expanding_learning_rate=1.0, expanding_learning_rate_decay=0.99,
                 expanding_smoothing_parameter=0.1, **kwargs):
        super().__init__(weights, winner_learning_rate, neighbours_learning_rate, **kwargs)
        self.expanding_learning_rate = expanding_learning_rate
        self.expanding_learning_rate_decay = expanding_learning_rate_decay
        self.smoothing_parameter = expanding_smoothing_parameter

    def _post_fit_one(self):
        super()._post_fit_one()
        if self.expanding_learning_rate > 0.001:
            for i, w in enumerate(self.neurons["value"]):
                raw_vector_difference = self.neurons["value"] - w
                affinity = 1.0/(self.smoothing_parameter + np.linalg.norm(self.neurons["value"] - w, axis=1))
                affinity[i] = 0.0
                neighbourhood_factor = np.repeat(affinity.reshape((affinity.shape[0], 1)),
                                                 raw_vector_difference.shape[1], axis=1)
                self.neurons["value"] += self.neighbours_learning_rate*np.multiply(neighbourhood_factor, raw_vector_difference)

            self.expanding_learning_rate *= self.expanding_learning_rate_decay



# distances_from_w = np.linalg.norm(W - W[largest_error_neuron_idx], axis=1)
# neighbourhood_factor = np.exp(-distances_from_w)
# distances_from_w[largest_error_neuron_idx] = np.inf
# neighbourhood_factor[largest_error_neuron_idx] = 0
# neighbours = neighbourhood_factor > 0.05
# if any(neighbours):
#     max_error, largest_error_neighbour_neuron_idx = 0.0, 0
#     for i, (record, is_neighbour) in enumerate(zip(self.neurons, neighbours)):
#         if is_neighbour and record["error"] > max_error:
#             max_error = record["error"]
#             largest_error_neighbour_neuron_idx = i
# else:
#     print("WARNING: No neighbours")
#     largest_error_neighbour_neuron_idx = np.argmin(distances_from_w)