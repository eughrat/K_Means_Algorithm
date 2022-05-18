import numpy as np
import random


def scaler(data):
    normalized_data = (data - min(data)) / (max(data) - min(data))
    return normalized_data


def random_centroids(data):
    centroids = [random.choice(data)]
    return centroids


def distance_to_centroids(centroid, datapoint):
    distance = np.sqrt(np.sum((centroid - datapoint) ** 2, axis=1))
    return distance
