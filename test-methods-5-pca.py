from modules import *
import json

# set constants
m = 100  # num of samples per cohort
cohorts_values = [2, 3, 4, 5]

total_data = {
    "m": m,
    "data": []
}

for cohorts in cohorts_values:
    file_path = 'multi-samples.json'
    data = GLV.generate_models(m, cohorts, file_path, force=True)

    states = []
    for i in range(cohorts):
        states.extend(data["models"][i]["cohort"])

    total_data["data"].append({
        "cohorts": cohorts,
        "points": states
    })

file_path = 'test_results-5-pca.json'
with open(file_path, 'w') as outfile:
    json.dump(total_data, outfile)
