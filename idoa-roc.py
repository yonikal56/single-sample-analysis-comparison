from modules import *
import numpy as np

# set constants
m = 100
num_of_samples = 100

# create two different GLV models with m samples
file_path = 'sample3.json'
data = GLV.generate_models(m, 1, file_path)

# create IDOA method class
idoa = IDOA.IDOA(data)
roc = ROC.ROC()

for i in range(1, 6):
    model = GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A'])
    samples, real = GLV.generate_random_shuffled_samples(data['models'][0], num_of_samples)
    pred = idoa.predict_real(data['models'][0]['cohort'], np.array(samples))

    roc.add_graph(real, pred, f'IDOA{i}')

roc.show()
