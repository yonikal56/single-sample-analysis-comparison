from modules import *
import numpy as np
import json

all_results = []

cohorts = 2
num_of_runs = 4
m = 100
num_of_samples = 100
delta_values = np.linspace(0, 0.5, 20)
file_path = 'noise_tests_samples.json'


for delta in delta_values:
    tests_results = []
    for nummm in range(num_of_runs):
        print(f'delta: {delta}, iteration number: {nummm}')
        tests_results.append(Testing.Testing.run_test(file_path, cohorts, m, num_of_samples, delta=delta))
    all_results += tests_results
file_path = 'noise_tests_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)

print("all done|!!!")
