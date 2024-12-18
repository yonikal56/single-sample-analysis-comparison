from modules import *
import numpy as np
import json

all_results = []

cohorts = 2
num_of_runs = 4
m_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]
num_of_samples = 100


file_path = 'binary_classification_samples.json'
for m in m_values:
    tests_results = []
    for _ in range(num_of_runs):
        tests_results.append(Testing.Testing.run_test(file_path, cohorts, m, num_of_samples))
    all_results += tests_results
file_path = 'binary_classification_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)

print("all done|!!!")
