import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data, no_of_centers):
        self.data = data
        self.no_of_centers = no_of_centers


    def get_centers_simple_method(self):
        centers = self.data.sample(self.no_of_centers)
        return centers

    def get_centers_uniform_method(self):
        min_, max_ = np.min(self.data, axis=0), np.max(self.data, axis=0)
        centers = pd.DataFrame([np.random.uniform(min_, max_) for _ in range(self.no_of_centers)], columns=self.data.columns)
        return centers

    def get_clusters_simple_method(self, centers):
        distances = np.zeros((len(self.data.index), self.no_of_centers))
        closest = np.argmin(distances, axis=1)

        while True:
            old_closest = closest.copy()

            for i in range(3):
                distances[:, i] = (((self.data.iloc[:, :] - centers.iloc[i, :]) ** 2).sum(axis=1)) ** 0.5
            closest = np.argmin(distances, axis=1)

            for i in range(len(centers.index)):
                centers.iloc[i] = self.data[closest == i].mean(axis=0)

            if all(closest == old_closest):
                break

            return closest, centers

    def get_clusters_iteration_method(self, centers, iteration=50):
        distances = np.zeros((len(self.data.index), self.no_of_centers))

        for _ in range(iteration):

            for i in range(3):
                distances[:, i] = (((self.data.iloc[:, :] - centers.iloc[i, :]) ** 2).sum(axis=1)) ** 0.5
            closest = np.argmin(distances, axis=1)

            for i in range(len(centers.index)):
                centers.iloc[i] = self.data[closest == i].mean(axis=0)

            return closest, centers


def scaler(data):
    return (data - data.min()) / (data.max() - data.min())

def get_2D_graph(data, centers, closest, x_name, y_name):
    plt.scatter(data[x_name], data[y_name], c=closest)
    plt.scatter(centers[x_name], centers[y_name], c='red')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def check_labels(clusters, labels,cluster_col_name,label_col_name):
    default_idx = [i for i in range(len(clusters))]
    df = pd.DataFrame(data=clusters, columns=[cluster_col_name], index=default_idx)
    df[label_col_name] = labels
    conditions = [df[cluster_col_name] == df[label_col_name],
                  df[cluster_col_name] != df[label_col_name]]
    choices = [int(1), int(0)]
    df['check'] = np.select(conditions, choices, default=1)

    return df[df['check'] == 1]['check'].sum() / len(df.index)
