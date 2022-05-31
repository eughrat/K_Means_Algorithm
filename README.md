# K-Means Clustering Algorithm

### PROJECT BRIEFING

Custom implementation of the K-Means clustering algorithm.

### PROJECT OVERVIEW

The purpose of K-Means is to assign data points to groups using parameters (can't be categorical) that describe the points. The method involves iteratively computing the distance between points and cluster centers. With each iteration, the cluster centers are newly selected. The iteration continues until no point changes the cluster it is in.

The whole method is strongly dependent on the selection of the first centers. This can affect the results of the calculation. Therefore, two methods for selecting the first cluster centers have been implemented:

- Random choice - in this approach, random data points are selected from the dataset and used as the initial centroids.

- K-Means++ -  the first centroid is randomly selected data point, next step is choosing the subsequent centroids from the remaining data points based on a probability proportional to the squared distance away from a given point's nearest existing centroid.

To reduce the dependence of the results on their size, the standard scaler function was used.

The following dataframes were used to validate the algorithm:

- Iris dataframe
- College_Student
- blob
- basic2
- basic4

### SUMMARY

For simple dataframes that contained little information (Iris, blob, basic4) about data points, the clustering results were good and largely matched the original splits.

For more complicated dataframes (basic2, College_Student) with a large number of parameters describing a data point or when the points are arranged in a pattern on the graph, no good clustering results were obtained.

### INSTALATION

Short guide to get startd with sofware:

1. Clone the repo or donwload zip file.
2. Open the folder with IDE editor.
3. Create conda virtual environment.
4. Install all the libraries form requirements.txt

### AUTHOR

Michał Piernicki 

