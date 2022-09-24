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
data = GLV.generate_models(m, cohorts, file_path)

graphs = graph.Graph(3)
# initialise the standard scaler
sc = StandardScaler()
# set the components to 2
pca = PCA(n_components=2, whiten=True)

states = []
for i in range(cohorts):
    states.extend(data['models'][i]['cohort'])

df = pd.DataFrame.from_dict(states)

df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)

pca.fit(df_scaled)

# fit the model to our data and extract the results
X_pca = pca.transform(df_scaled)

# create a dataframe from the dataset
df = pd.DataFrame(data=X_pca,
                  columns=["Component1",
                           "Component2"])

points = []
for i in range(cohorts):
    start = i * m
    end = (i + 1) * m - 1
    graphs.scatter(df['Component1'].iloc[start:end], df['Component2'].iloc[start:end], graphs.get_axes()[0], False, False)

for i in range(cohorts):
    distance = DistanceCheck.DistanceCheck.calculate_in_group_distance(data['models'][i]['cohort'])
    print(f'cohort {i} in group distance - {distance}')
for i in range(cohorts):
    for j in range(i+1, cohorts):
        distance = DistanceCheck.DistanceCheck.calculate_between_group_distance(data['models'][i]['cohort'],
                                                                                data['models'][j]['cohort'])
        print(f'distance between {i} and {j} cohorts - {distance}')

num_of_points = 10
all_vales = [np.linspace(0.1, 0.6, num_of_points), np.linspace(0.025, 0.05, num_of_points)]
for k in range(2):
    values = all_vales[k]
    group_distances = []
    mean_group_distances = []
    between_groups_distances = []
    group_distances_data = []
    num_of_samples = 50
    for value in values:
        in_distances = []
        cohorts = []
        for i in range(2):
            A = GLV.GLV.get_random_A(probability=value, bound=0.025) if k == 0 else GLV.GLV.get_random_A(bound=value, probability=0.1)
            model = GLV.GLV(r=data['models'][0]['r'], A=A)
            samples = [sample.tolist() for sample in model.get_samples(num_of_samples)]
            cohorts.append(samples)
            distance = DistanceCheck.DistanceCheck.calculate_in_group_distance(samples)
            in_distances.append(distance)
        between_groups_distances.append(DistanceCheck.DistanceCheck.calculate_between_group_distance(cohorts[0], cohorts[1]))
        datac = np.concatenate((in_distances, [np.array(in_distances).mean()]))
        group_distances_data.append(datac)
    distance = all_vales[k][1] - all_vales[k][0]
    graphs.get_axes()[k+1].set_xlim(all_vales[k][0]-distance/2, all_vales[k][-1]+distance/2)
    graphs.boxplot(group_distances_data, graphs.get_axes()[k+1], positions=np.round(all_vales[k], 3), width=distance/2)
    graphs.plot(all_vales[k], between_groups_distances, graphs.get_axes()[k+1], color='green')
print("finished")

graphs.show()
