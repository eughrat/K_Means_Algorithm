import numpy as np
import pandas as pd
import seaborn as sns

from Algorithm import Kmeans, scaler, get_results


def while_method():
    df = pd.read_csv(r'C:\Users\mpiernicki\PycharmProjects\K_Means_Algorithm\DATA\Iris.csv')
    df.drop(['Id', 'Species'], axis=1, inplace=True)

    df_scale = scaler(df)
    kmeans = Kmeans(df_scale, 3)
    init_centers = kmeans.get_init_centers_simple_method()
    labels, centers = kmeans.get_final_centers_simple_method(init_centers)

    df_check = pd.read_csv(r'C:\Users\mpiernicki\PycharmProjects\K_Means_Algorithm\DATA\Iris.csv')
    sns.scatterplot(data=df_check, x='SepalLengthCm', y='SepalWidthCm', hue='Species')

    df_scale['Cluster'] = labels
    df_scale['Cluster'].value_counts()
    df_scale['Cluster'] = df_scale['Cluster'].apply(lambda x: ('Iris-setosa' if x == 0
                                                               else 'Iris-virginica' if x == 1
    else 'Iris-versicolor'))

    df_scale['Species'] = df_check['Species']
    conditions = [df_scale['Cluster'] == df_scale['Species'],
                  df_scale['Cluster'] != df_scale['Species']]
    choices = [int(1), int(0)]
    df_scale['check'] = np.select(conditions, choices, default=1)
    ratio = df_scale[df_scale['check'] == 1]['check'].sum() / len(df_scale.index)

    return ratio


get_results(while_method,'while_method')

def loop_method():
    df = pd.read_csv(r'C:\Users\mpiernicki\PycharmProjects\K_Means_Algorithm\DATA\Iris.csv')
    df.drop(['Id', 'Species'], axis=1, inplace=True)

    df_scale = scaler(df)
    kmeans = Kmeans(df_scale, 3)
    init_centers = kmeans.get_init_centers_simple_method()
    labels, centers = kmeans.get_final_centers_iteration_method(init_centers)

    df_check = pd.read_csv(r'C:\Users\mpiernicki\PycharmProjects\K_Means_Algorithm\DATA\Iris.csv')
    sns.scatterplot(data=df_check, x='SepalLengthCm', y='SepalWidthCm', hue='Species')

    df_scale['Cluster'] = labels
    df_scale['Cluster'].value_counts()
    df_scale['Cluster'] = df_scale['Cluster'].apply(lambda x: ('Iris-setosa' if x == 0
                                                               else 'Iris-virginica' if x == 1
    else 'Iris-versicolor'))

    df_scale['Species'] = df_check['Species']
    conditions = [df_scale['Cluster'] == df_scale['Species'],
                  df_scale['Cluster'] != df_scale['Species']]
    choices = [int(1), int(0)]
    df_scale['check'] = np.select(conditions, choices, default=1)
    ratio = df_scale[df_scale['check'] == 1]['check'].sum() / len(df_scale.index)

    return ratio

get_results(loop_method,'loop_method')


