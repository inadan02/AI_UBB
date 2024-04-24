import numpy as np
from numpy.random import uniform
import random


def euclidean(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class MyKNN:
    def __init__(self, n_clusters=10, max_iterations=1000):
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, inputs):
        self.centroids = [random.choice(inputs)]
        for _ in range(self.n_clusters - 1):
            distances = np.sum([euclidean(centroid, inputs) for centroid in self.centroids], axis=0)
            # calculate distances from points to the centroids
            distances /= np.sum(
                distances)  # this line normalizes the distances by dividing each distance by the sum of all distances. This step ensures that the normalized distances represent probabilities.
            new_centroid_index, = np.random.choice(range(len(inputs)), size=1,
                                                   p=distances)  # select a new centroid based on the probabilities defined by the normalized distances
            # choose remaining points based on their distances
            self.centroids += [inputs[new_centroid_index]]

        iteration = 0
        previous_centroids = None
        while np.not_equal(self.centroids,
                           previous_centroids).any() and iteration < self.max_iterations:  # the centroids stop changing or the max number of iterations is reached
            sorted_points = [[] for _ in range(self.n_clusters)]  # assign each data to the nearest centroid
            for input in inputs:
                distances = euclidean(input, self.centroids)
                centroid_index = np.argmin(distances)  # choose the minimum distance, the cluster input is assigned to
                sorted_points[centroid_index].append(input)
            previous_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            # reassign centroids as mean of the points belonging to them

            for i, centroid in enumerate(self.centroids):
                # if any centroid contains NaN values, it is replaced with the corresponding centroid from the previous iteration
                if np.isnan(centroid).any():
                    self.centroids[i] = previous_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_indexes = []
        for x in X:
            distances = euclidean(x, self.centroids)
            centroid_index = np.argmin(distances)
            centroids.append(self.centroids[centroid_index])
            centroid_indexes.append(centroid_index)
        return centroids, centroid_indexes
