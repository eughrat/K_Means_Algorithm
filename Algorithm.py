import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


class Kmeans:
    """
    This class is used to divide clusters using K-means methods, initially takes the following parameters:
    * dataframe: pd.DataFrame
    * number of clusters: int
    """

    def __init__(self, data, no_of_centers):
        self.data = data
        self.no_of_centers = no_of_centers

    def get_init_centers(self):
        """
        This function returns the initial cluster centers using the classic K-Means method
        """

        "Random choice of centers"
        centers = self.data.sample(self.no_of_centers)

        return centers

    def get_init_centers_plusplus(self):
        """
        This function returns the initial cluster centers using the K-Means++ method
        """
        
        "To avoid data leakage copy of df is made" 
        init_df = self.data.copy()
        
        "First center is choosen randomly, and is dropped from df"
        centers = pd.DataFrame(init_df.sample(1))
        init_df.drop(centers.iloc[[0]].index)

        "Choice of the rest of initial centers"
        for _ in range(self.no_of_centers - 1):

            "distances will be used as a storage for distances between points and initial centers"
            distances = []
            for i in range(len(init_df)):

                "point is storage for parameters of each point in dataframe"
                point = init_df.iloc[[i]].values
                
                "d will be used for calculation of minimal distances between initial centers and points that are temporary assigned to centers"
                d = sys.maxsize

                "For each point we calculate the minimum distance between initial centers and we assign point to one of them"
                for j in range(len(centers)):
                    init_distance = ((point - centers.iloc[[j]].values) ** 2).sum(axis=1)
                    d = min(d, init_distance)

                distances.append(d)
                
            "Here we assign the point farthest from the rest of the centers to the initial centers storage (centers)"
            distances = np.array(distances)
            next_center = init_df.iloc[[np.argmax(distances)]]
            centers.loc[len(centers.index)] = next_center.iloc[0, :]
            init_df.drop(next_center.iloc[[0]].index, inplace=True)

        return centers

    def get_final_centers(self, centers):
        """
        This function returns the final centers of clusters and the labels of points.
        It iteratively checks the distances between points and centers by assigning points to the nearest centers, until the labels do not change.
        Initially takes the following parameters:
        * centers: pd.DataFrame (a dataframe that contains the initial cluster centers)
        """
        
        "distances is storage for distance from centers to each point"
        distances = np.zeros((len(self.data.index), self.no_of_centers))
        "closest is storage for closest points from each center"
        closest = np.zeros(len(self.data)).astype(int)

        while True:
            
            "The while loop end when closest points from the centers dont change their labels, so we need to make copy of closest to compare it at the and of the loop"
            old_closest = closest.copy()

            "Simple calculation of euclidan distance between points and centers" 
            for i in range(len(centers)):
                distances[:, i] = (((self.data.iloc[:, :] - centers.iloc[i, :]) ** 2).sum(axis=1)) ** 0.5
            closest = np.argmin(distances, axis=1)

            "Calculation of new centers using the average distance between points in a cluster"
            for i in range(len(centers.index)):
                centers.iloc[i] = self.data[closest == i].mean(axis=0)
                
            "checking if there were any changes of points between clusters"
            if all(closest == old_closest):
                break

            return closest, centers


def scaler(data):
    """
    Standardize features by removing the mean and scaling to unit variance.
    Initially takes the following parameters:
    * data: pd.DataFrame (a dataframe that contains the initial cluster centers)
    """

    return (data - data.std()) / data.mean()


