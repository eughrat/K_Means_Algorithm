import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data, no_of_centers):
        self.data = data
        self.no_of_centers = no_of_centers

    def get_init_centers_simple_method(self):
        # shuffled_data = self.data.sample(frac=1).reset_index(drop=True)
        centers = self.data.sample(self.no_of_centers)
        return centers

    def get_init_centers_uniform_method(self):
        shuffled_data = self.data.sample(frac=1).reset_index(drop=True)
        min_, max_ = np.min(shuffled_data, axis=0), np.max(shuffled_data, axis=0)
        centers = pd.DataFrame([np.random.uniform(min_, max_) for _ in range(self.no_of_centers)], columns=shuffled_data.columns)
        return centers

    def get_final_centers_simple_method(self, centers):
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

    def get_final_centers_iteration_method(self, centers, iteration=50):
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


def get_results(method, file_name='results'):
    results = []
    for _ in range(100):
        results.append(method())

    print(results)
    print(np.mean(results))
    default_idx = [i for i in range(len(results))]
    df_loop_results = pd.DataFrame(results, index=default_idx, columns=[file_name])

    return df_loop_results.to_csv(file_name)

#
# def check_labels(centers, labels, center_col_name = 'centers', label_col_name = 'labels'):
#     default_idx = [i for i in range(len(centers))]
#     df = pd.DataFrame(data=centers, columns=[center_col_name], index=default_idx)
#     df[label_col_name] = labels
#     conditions = [df[center_col_name] == df[label_col_name],
#                   df[center_col_name] != df[label_col_name]]
#     choices = [int(1), int(0)]
#     df['check'] = np.select(conditions, choices, default=1)
#
#     return df[df['check'] == 1]['check'].sum() / len(df.index)
