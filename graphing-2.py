from modules import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


# set constants
m = 100  # num of samples per cohort
cohorts = 2

def plot_cohorts_pca_distances(cohort_a, cohort_b, title, order=True):
    if order is True:
        graphs = graph.Graph(2, title=title)
    else:
        graphs = graph.Graph(1,2, title=title)
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

    graphs.scatter(df['Component1'].iloc[0:m-1], df['Component2'].iloc[0:m-1], graphs.get_axes()[0], color="blue", size=5, label="Cohort A")
    graphs.scatter(df['Component1'].iloc[m:2*m-1], df['Component2'].iloc[m:2*m-1], graphs.get_axes()[0], color="red", size=5, label="Cohort B")
    graphs.get_axes()[0].set_xlabel('PC1')
    graphs.get_axes()[0].set_ylabel('PC2')
    graphs.get_axes()[0].legend()
    graphs.get_axes()[1].set_xlabel('Bray–Curtis dissimilarity')
    graphs.get_axes()[1].set_ylabel('Probability')
    graphs.get_axes()[1].set_yticklabels([])

    distances = []
    for i in range(len(cohort_a)):
        for j in range(i + 1, len(cohort_a)):
            distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_a[i], cohort_a[j], 0))
    graphs.hist(distances, graphs.get_axes()[1], "blue", label="A-A")
    distances = []
    for i in range(len(cohort_b)):
        for j in range(i + 1, len(cohort_b)):
            distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_b[i], cohort_b[j], 0))
    graphs.hist(distances, graphs.get_axes()[1], "red", label="B-B")

    distances = []
    for i in range(len(cohort_a)):
        for j in range(len(cohort_b)):
            distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_a[i], cohort_b[j], 0))
    graphs.hist(distances, graphs.get_axes()[1], "green", label="A-B")

    graphs.legend()
    graphs.show()

def plot_cohorts_pca_distances_all(cohorts_arr, titles):
    graphs = graph.Graph(2, len(cohorts_arr))

    nn = 0
    for cohort_a, cohort_b in cohorts_arr:
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

        ax1 = graphs.get_axes()[0][nn]
        ax2 = graphs.get_axes()[1][nn]
        graphs.scatter(df['Component1'].iloc[0:m-1], df['Component2'].iloc[0:m-1], ax1, color="blue", size=5, label="Cohort A")
        graphs.scatter(df['Component1'].iloc[m:2*m-1], df['Component2'].iloc[m:2*m-1], ax1, color="red", size=5, label="Cohort B")

        distances = []
        for i in range(len(cohort_a)):
            for j in range(i + 1, len(cohort_a)):
                distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_a[i], cohort_a[j], 0))
        graphs.hist(distances, ax2, "blue", label="A-A")
        distances = []
        for i in range(len(cohort_b)):
            for j in range(i + 1, len(cohort_b)):
                distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_b[i], cohort_b[j], 0))
        graphs.hist(distances, ax2, "red", label="B-B")

        distances = []
        for i in range(len(cohort_a)):
            for j in range(len(cohort_b)):
                distances.append(DistanceCheck.DistanceCheck.calculate_distance(cohort_a[i], cohort_b[j], 0))
        graphs.hist(distances, ax2, "green", label="A-B")

        ax1.set_title(titles[nn], loc='left', fontsize=15, weight='bold', x=-0.1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])
        if nn == 0:
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            ax1.legend()
            ax2.set_xlabel('Bray–Curtis dissimilarity')
            ax2.set_ylabel('Probability')
            ax2.legend()
        else:
            ax1.set_xlabel('')

        nn += 1

    graphs.show()

file_path = 'samples-1-dist.json'
# data = GLV.generate_models(m, cohorts, file_path, force=False, sameR=True, bound=0.025)
# plot_cohorts_pca_distances(data['models'][0]['cohort'], data['models'][1]['cohort'], "graph 1", False)
data1 = GLV.generate_models(m, cohorts, 'samples-2-1.json', force=False, sameR=False, bound=0.025)
data2 = GLV.generate_models(m, cohorts, 'samples-2-2.json', force=False, sameR=True, bound=0.1)
data3 = GLV.generate_models(m, cohorts, 'samples-2-3.json', force=False, sameR=True, bound=0.04)
data4 = GLV.generate_models(m, cohorts, 'samples-2-4.json', force=True, sameR=True, bound=0.025)
data5 = GLV.generate_models(m, cohorts, 'samples-2-5.json', force=False, sameR=True, bound=0.015)
data = [data1, data2, data3, data4, data5]
plot_cohorts_pca_distances_all([(one_dataset['models'][0]['cohort'], one_dataset['models'][1]['cohort']) for one_dataset in data], ['a', 'b', 'c', 'd', 'e'])
