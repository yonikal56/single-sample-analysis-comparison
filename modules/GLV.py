import random
import numpy as np
from scipy.integrate import odeint
import json
import os.path
from modules.progressbar import ProgressBar


class GLV:
    numOfPopulations = 100

    @staticmethod
    def decision(probability):
        return random.random() < probability

    @staticmethod
    def model(X, t, A, r):
        Amat = np.array(A)
        Xvec = np.array(X)
        rVec = np.array(r)
        fVec = rVec + Amat.dot(Xvec)
        return np.array([Xvec[j] * fVec[j] for j in range(len(r))])

    @staticmethod
    def get_random_A(bound=0.025, probability=0.1):
        Arandoms = np.random.uniform(-bound, bound, GLV.numOfPopulations * (GLV.numOfPopulations - 1))
        A = []
        for i in range(GLV.numOfPopulations):
            A.append([])
            for j in range(GLV.numOfPopulations):
                if i == j:
                    A[i].append(-1)
                else:
                    A[i].append(Arandoms[0] if GLV.decision(probability) else 0)
                    Arandoms = np.delete(Arandoms, 0)
        return A

    @staticmethod
    def get_random_r(lower_limit=0, upper_limit=1):
        return np.random.uniform(lower_limit, upper_limit, GLV.numOfPopulations)

    def get_random_initials(self, lower_limit=0, upper_limit=1):
        prob = np.random.uniform(0.6, 0.9, 1)
        full_initials = list(np.random.uniform(lower_limit, upper_limit, GLV.numOfPopulations))
        return [initial if GLV.decision(prob) else 0 for initial in full_initials]

    def solve_model(self, initials, time=50, time_fractions=5):
        t = np.linspace(0, time, time_fractions)
        return odeint(GLV.model, initials, t, args=(self.__A, self.__r)).T

    def get_sample(self):
        initials = self.get_random_initials()
        data = self.solve_model(initials)
        populations = np.array([a[-1] for a in data])
        return populations / sum(populations)

    def get_samples(self, m):
        samples = []
        progress = ProgressBar(m, f'creating {str(m)} samples', 25)
        for i in range(m):
            samples.append(self.get_sample())
            progress.update()
        del progress
        return samples

    def get_A(self):
        return self.__A

    def get_r(self):
        return self.__r

    def __init__(self, r=None, A=None):
        self.__r = r if r is not None else GLV.get_random_r()
        self.__A = A if A is not None else GLV.get_random_A()


def generate_models(m, cohorts, file_path, bound=0.025, probability=0.1, force=False):
    data = {}
    if os.path.exists(file_path):
        with open(file_path) as file:
            data = json.load(file)
        run_again = len(data["models"]) != cohorts
    else:
        run_again = True

    if run_again or force:
        data = {}
        models = []
        r = GLV.get_random_r()
        for n in range(cohorts):
            A = GLV.get_random_A(probability=probability, bound=bound)
            model = GLV(r=r, A=A)
            samples = [sample.tolist() for sample in model.get_samples(m)]
            models.append({
                "A": model.get_A(),
                "cohort": samples,
                "index": n,
                "r": r.tolist()
            })
        data['models'] = models
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)
    return data


def generate_random_samples(data, num):
    samples = []
    real = []
    progress = ProgressBar(num, f'creating {str(num)} samples', 25)
    for i in range(num):
        chosen_model = random.choice(data['models'])
        model = GLV(r=chosen_model['r'], A=chosen_model['A'])
        samples.append(model.get_sample())
        real.append(chosen_model['index'])
        progress.update()
    del progress
    return samples, real
