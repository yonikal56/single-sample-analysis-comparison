from modules import *
import numpy as np

# set constants
m = 50  # num of samples per cohort
cohorts = 3

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(m, cohorts, file_path)

# create IDOA method class
idoa = IDOA.IDOA(data)

samples, real = GLV.generate_random_samples(data, 1)
sample = samples[0]

graphs = graph.Graph(2)
for n in range(cohorts):
    idoa_value = idoa.calculate_IDOA(data['models'][n]["cohort"], sample, False)

print(real)

# predictions
samples, real = GLV.generate_random_samples(data, 100)
predictions = idoa.predict(samples, True)
real_idoa_values = []
wrong_idoa_values = []
results = []
num_of_success = 0
for prediction, re in zip(predictions, real):
    idoa_values = prediction[1]
    pred = prediction[0]
    if pred == re:
        num_of_success += 1
        real_idoa_values.append(idoa_values[pred])
        wrong_idoa_values.extend([idoa_values[n] for n in range(cohorts) if n != pred])
    results.append((pred, re, re == pred))
print(results)
print(f'success_rate : {(num_of_success / len(real)) * 100}')
print('real values mean:' + str(np.array(real_idoa_values).mean()))
print('wrong values mean:' + str(np.array(wrong_idoa_values).mean()))
graphs.scatter(real_idoa_values, np.linspace(0, 1, len(real_idoa_values)), graphs.get_axes()[0], False, False)
graphs.plot([np.array(real_idoa_values).mean()]*2, np.linspace(0, 1, 2), graphs.get_axes()[0], line_style='-', set_y_lim=False, color='green')
flatten_wrongs = np.array(wrong_idoa_values).flatten()
graphs.scatter(flatten_wrongs, np.linspace(0, 1, len(flatten_wrongs)), graphs.get_axes()[0], False, False)
graphs.plot([np.array(flatten_wrongs).mean()]*2, np.linspace(0, 1, 2), graphs.get_axes()[0], line_style='-', set_y_lim=False)
graphs.hist(real_idoa_values, graphs.get_axes()[1])
graphs.hist(wrong_idoa_values, graphs.get_axes()[1])

graphs.show()
