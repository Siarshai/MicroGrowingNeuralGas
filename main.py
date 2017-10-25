from datagen import generate_two_simple_blobs, generate_line_inside_circle, generate_five_hills_in_ring, generate_donuts, \
    generate_two_blobs_with_outliers, generate_blob_inside_blob, create_dynamic_two_simple_blobs_fn
from ergng import ExpandingSRGNG
from gng import GrowingNeuralGas, HistoryLevel
import numpy as np
from rgng import QinRobustGrowingNeuralGas, SimpleRobustGrowingNeuralGas
from visualization import render_gng_animation, render_scatter_gng

np.random.seed(1337)

# X = generate_two_blobs_with_outliers(number_of_outliers=10, noise_multiplyer=10.0)
# X = generate_five_hills_in_ring(size_of_cluster=100)
# X = generate_blob_inside_blob()
# X = 5*generate_line_inside_circle(size_of_cluster=150)
X = 2*generate_donuts(size_of_cluster=75, lattice_width=2, lattice_height=2)


for clusterizer, label in zip(
    [GrowingNeuralGas], ["GNG"]
):
    gng = clusterizer(np.asarray([[0.0, 0.0], [1.0, 1.0]]),
                        winner_learning_rate=0.2,
                        neighbours_learning_rate=0.01,
                        learning_rate_decay=1.0,
                        edge_max_age=5,
                        populate_iterations_divisor=100,
                        max_neurons=50,
                        insertion_error_decay=0.8,
                        iteration_error_decay=0.99,
                        with_history=HistoryLevel.EPOCHS,
                        force_dying=False)

    epoch_stride = 20
    gng.fit(X, epoch_stride)
    W = gng.get_weights()
    C = gng.get_connections_idx_pairs()
    render_scatter_gng(W, X, C, title=label + " epoch " + str(epoch_stride), show=False)

for clusterizer, label in zip(
    [GrowingNeuralGas], ["GNG"]
):
    gng = clusterizer(np.asarray([[0.0, 0.0], [1.0, 1.0]]),
                        winner_learning_rate=0.2,
                        neighbours_learning_rate=0.01,
                        learning_rate_decay=1.0,
                        edge_max_age=50,
                        populate_iterations_divisor=100,
                        max_neurons=50,
                        insertion_error_decay=0.8,
                        iteration_error_decay=0.99,
                        with_history=HistoryLevel.EPOCHS,
                        force_dying=True)
    gng.fit_p(create_dynamic_two_simple_blobs_fn(border_freq=0.005))
    weights_history = gng.get_weights_history()
    connections_history = gng.get_connections_history()
    samples_history = gng.get_samples_history()
    render_gng_animation(weights_history, X=None, connections_history=connections_history,
                         samples_history=samples_history,
                         title="GNG Fitting", show=False, frame_divisor=50)
