from modules import *
import numpy as np
import json

all_results = []

cohorts_values = list(range(2,6))
num_of_runs = 4
m = 100
num_of_samples = 100
file_path = 'multiclass_tests_samples.json'

for cohorts in cohorts_values:
    tests_results = []
    for _ in range(num_of_runs):
        tests_results.append(Testing.Testing.run_test(file_path, cohorts, m, num_of_samples))
    all_results += tests_results
file_path = 'multiclass_tests_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)

print("all done|!!!")
