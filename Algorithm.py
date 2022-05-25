import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data, no_of_centers):
        self.data = data
        self.no_of_centers = no_of_centers

    def get_init_centers(self):
        centers = self.data.sample(self.no_of_centers)

        return centers

    def get_init_centers_plusplus(self):
        centers = pd.DataFrame(self.data.sample(1))

        for _ in range(self.no_of_centers - 1):

            for j in range(len(centers)):
                init_distances = ((self.data.values - centers.iloc[[j]].values) ** 2).sum(axis=1)

            next_center = self.data.iloc[[np.argmax(init_distances)]]

            centers.loc[len(centers.index)] = next_center.iloc[0,:]

        return centers

    def get_final_centers(self, centers):
        distances = np.zeros((len(self.data.index), self.no_of_centers))
        closest = np.zeros(len(self.data)).astype(int)

        while True:
            old_closest = closest.copy()

            for i in range(len(centers)):
                distances[:, i] = (((self.data.iloc[:, :] - centers.iloc[i, :]) ** 2).sum(axis=1)) ** 0.5
            closest = np.argmin(distances, axis=1)

            for i in range(len(centers.index)):
                centers.iloc[i] = self.data[closest == i].mean(axis=0)

            if all(closest == old_closest):
                break

            return closest, centers


def scaler(data):
    return (data - data.min()) / (data.max() - data.min())


def get_2D_graph(data, centers, closest, x_name, y_name):
    plt.scatter(data[x_name], data[y_name], c=closest)
    plt.scatter(centers[x_name], centers[y_name], c='red')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def get_results(method, file_name='results'):
    results = []
    for _ in range(100):
        results.append(method())

    print(results)
    print(np.mean(results))
    default_idx = [i for i in range(len(results))]
    df_loop_results = pd.DataFrame(results, index=default_idx, columns=[file_name])

    return df_loop_results.to_csv(file_name)
