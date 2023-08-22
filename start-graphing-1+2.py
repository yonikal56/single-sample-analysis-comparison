from modules import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# set constants
m = 100  # num of samples per cohort
cohorts = 2

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(m, cohorts, file_path, force=True, sameR=True, bound=0.1)


def plot_cohorts_pca_distances(cohort_a, cohort_b, m):
    graphs = graph.Graph(1,2)
    # initialise the standard scaler
    sc = StandardScaler()
    # set the components to 2
    pca = PCA(n_components=2, whiten=True)

    states = []
    states.extend(cohort_a)
    states.extend(cohort_b)

    df = pd.DataFrame.from_dict(states)

    df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)

    pca.fit(df_scaled)

    # fit the model to our data and extract the results
    X_pca = pca.transform(df_scaled)

    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_pca,
                      columns=["Component1",
                               "Component2"])

    graphs.scatter(df['Component1'].iloc[0:m-1], df['Component2'].iloc[0:m-1], graphs.get_axes()[0], False, False, color="blue")
    graphs.scatter(df['Component1'].iloc[m:2*m-1], df['Component2'].iloc[m:2*m-1], graphs.get_axes()[0], False, False, color="red")

    distances = []
    for i in range(len(cohort_a)):
        for j in range(i + 1, len(cohort_a)):
            distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_a[i], cohort_a[j], 0))
    graphs.hist(distances, graphs.get_axes()[1], "blue")
    distances = []
    for i in range(len(cohort_b)):
        for j in range(i + 1, len(cohort_b)):
            distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_b[i], cohort_b[j], 0))
    graphs.hist(distances, graphs.get_axes()[1], "red")

    distances = []
    for i in range(len(cohort_a)):
        for j in range(len(cohort_b)):
            distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_a[i], cohort_b[j], 0))
    graphs.hist(distances, graphs.get_axes()[1], "green")

    graphs.show()


plot_cohorts_pca_distances(data['models'][0]['cohort'], data['models'][1]['cohort'], m)
