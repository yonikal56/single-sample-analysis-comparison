from modules import *
import numpy as np
import json

all_results = []


cohorts = 2
num_of_runs = 4
m = 100
delta_values = np.linspace(0, 1, 20)[::-1]
num_of_samples = 100


file_path = 'gradual_classification_samples.json'
for k in range(num_of_runs):
    print(f'run number {k+1} from {num_of_runs}:')
    tests_results = [Testing.Testing.run_gradual_change_test(file_path, m, num_of_samples, delta_values)]
    print(tests_results)
    all_results.append(tests_results)
file_path = 'gradual_classification_results.json'
print(all_results)
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)

print("all done|!!!")
