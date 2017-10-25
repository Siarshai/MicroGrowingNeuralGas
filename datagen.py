import scipy
import numpy as np

def create_dynamic_two_simple_blobs_fn(border_mean=3.0, border_amplitude=3.0, border_freq=0.01):
    def dynamic_two_simple_blobs_fn(t, n):
        return generate_two_simple_blobs(n//2, border_width=border_mean+border_amplitude*np.sin(border_freq*t))
    return dynamic_two_simple_blobs_fn

def generate_two_simple_blobs(size_of_cluster=50, skew_factor=1.0, scatter_factor=1.0, border_width=2.0):
    X = scipy.random.standard_normal((size_of_cluster, 2)) + border_width
    X = np.concatenate((X, scatter_factor*(scipy.random.standard_normal((int(skew_factor*size_of_cluster), 2)) - border_width)))
    return X

def generate_two_blobs_with_outliers(size_of_cluster=75,
            skew_factor=1.0, scatter_factor=1.0, border_width=2.0,
            number_of_outliers=10, noise_multiplyer=10.0, noise_asymmetrcity=0.0):
    X = generate_two_simple_blobs(size_of_cluster, skew_factor, scatter_factor, border_width)
    O = np.random.rand(number_of_outliers, 2)
    O -= 0.5
    O[:, 0] -= noise_asymmetrcity
    O *= scatter_factor + noise_multiplyer*border_width
    X = np.concatenate([X, O])
    return X

def generate_blob_inside_blob(size_of_cluster=150,
            shrink_factor=0.5):
    X = scipy.random.standard_normal((size_of_cluster, 2))
    X = np.concatenate((X, shrink_factor*scipy.random.uniform(size=(size_of_cluster, 2))))
    return X


def generate_five_hills_in_ring(size_of_cluster=80):

    x1_covariance_matrix = [[0.25, 0], [0.1, 0.5]]
    x1_mean = (1, 2)
    x2_covariance_matrix = [[0.5, 1.25], [-0.2, -0.25]]
    x2_mean = (0, -2)
    x3_covariance_matrix = [[0.7, 0.15], [0.15, 0.2]]
    x3_mean = (-3, -3)
    x4_covariance_matrix = [[0.5, -4.75], [0.2, -0.25]]
    x4_mean = (-3, 3)
    x5_covariance_matrix = [[0.15, -1.75], [0.1, -0.1]]
    x5_mean = (0, 6)

    X1 = np.random.multivariate_normal(x1_mean, x1_covariance_matrix, (size_of_cluster))
    X2 = np.random.multivariate_normal(x2_mean, x2_covariance_matrix, (size_of_cluster))
    X3 = np.random.multivariate_normal(x3_mean, x3_covariance_matrix, (size_of_cluster))
    X4 = np.random.multivariate_normal(x4_mean, x4_covariance_matrix, (2*size_of_cluster))
    X5 = np.random.multivariate_normal(x5_mean, x5_covariance_matrix, (3*size_of_cluster//2))
    X = np.concatenate((X1, X2, X3, X4, X5))

    return X


def generate_line_inside_circle(size_of_cluster=100, circle_radius=1.0):

    X1 = np.random.rand(size_of_cluster, 2)*circle_radius/5.0
    random_angles = 1337*np.random.random(size=size_of_cluster)
    X1[:, 0] += circle_radius*np.sin(random_angles)
    X1[:, 1] += circle_radius*np.cos(random_angles)

    X2 = 2*circle_radius*(np.random.rand(size_of_cluster, 2) - 0.5)
    X2[:, 0] /= 20

    X = np.concatenate((X1, X2))
    return X


def generate_donuts(size_of_cluster=50, circle_radius=1.0, distance=2.0, lattice_width=2, lattice_height=2):
    D = []
    for x in range(lattice_width):
        for y in range(lattice_height):
            X = np.random.rand(size_of_cluster, 2)*circle_radius/5.0
            random_angles = 1337*np.random.random(size=size_of_cluster)
            X[:, 0] += circle_radius*np.sin(random_angles) + distance*x
            X[:, 1] += circle_radius*np.cos(random_angles) + distance*y
            D.append(X)
    return np.concatenate(D)