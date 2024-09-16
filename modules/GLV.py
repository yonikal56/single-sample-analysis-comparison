import random
import numpy as np
from scipy.integrate import odeint
import json
import os.path
from modules.progressbar import ProgressBar


class GLV:
    numOfPopulations = 100
    delta = None

    @staticmethod
    def decision(probability):
        # get True/False value with probability
        return random.random() < probability

    @staticmethod
    def model(X, t, A, r):
        # the GLV model equation
        Amat = np.array(A)
        Xvec = np.array(X)
        rVec = np.array(r)
        fVec = rVec + Amat.dot(Xvec)
        return np.array([Xvec[j] * fVec[j] for j in range(len(r))])

    @staticmethod
    def get_random_A(bound=0.025, probability=0.1):
        # return random interaction matrix (A)
        # all values are from uniform distribution between -bound and bound
        Arandoms = np.random.uniform(-bound, bound, GLV.numOfPopulations * (GLV.numOfPopulations - 1))
        A = []
        for i in range(GLV.numOfPopulations):
            A.append([])
            for j in range(GLV.numOfPopulations):
                if i == j:
                    # -1 in main diagonal
                    A[i].append(-1)
                else:
                    # probability of interaction
                    A[i].append(Arandoms[0] if GLV.decision(probability) else 0)
                    Arandoms = np.delete(Arandoms, 0)
        return A

    @staticmethod
    def get_random_r(lower_limit=0, upper_limit=1):
        # return random r vector using uniform distribution between two limits
        return np.random.uniform(lower_limit, upper_limit, GLV.numOfPopulations)

    def get_random_initials(self, lower_limit=0, upper_limit=1):
        # get random initial values
        # probability of value to be not 0
        prob = np.random.uniform(0.6, 0.9, 1)
        # each value (that is not 0) is from uniform distribution between lower and upper limits
        full_initials = list(np.random.uniform(lower_limit, upper_limit, GLV.numOfPopulations))
        return [initial if GLV.decision(prob) else 0 for initial in full_initials]

    def solve_model(self, initials, time=50, time_fractions=5):
        t = np.linspace(0, time, time_fractions)
        # add noise
        noisyB = self.get_random_A()
        if self.delta is None:
            delta = np.random.uniform(0, 0.3)
        else:
            delta = self.delta
        noisyA = (1 - delta) * np.array(self.__A) + delta * np.array(noisyB)
        # solve GLV equations
        return odeint(GLV.model, initials, t, args=(noisyA, self.__r)).T

    def get_shuffled_sample(self, cohort):
        # copy cohort
        samples = cohort.copy()
        # each index is shuffled sample is a value from random sample inside cohort in the same index
        sample = np.array([random.choice(samples)[i] for i in range(GLV.numOfPopulations)])
        return sample

    def get_shuffled_samples(self, m, cohort):
        # create m shuffled samples
        samples = []
        progress = ProgressBar(m, f'creating {str(m)} shuffled samples', 25)
        for i in range(m):
            samples.append(self.get_shuffled_sample(cohort))
            progress.update()
        del progress
        return samples

    def get_sample(self):
        # get real sample
        initials = self.get_random_initials()
        data = self.solve_model(initials)
        populations = np.array([a[-1] for a in data])
        #return populations  # use for unsupervised classification
        return populations / sum(populations)

    def get_samples(self, m):
        # get m real samples
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


def generate_models(m, cohorts, file_path, bound=0.025, probability=0.1, force=False, sameR=True):
    # generate cohorts
    # check if file does not exist or exist with different number of cohorts
    data = {}
    if os.path.exists(file_path):
        with open(file_path) as file:
            data = json.load(file)
        run_again = len(data["models"]) != cohorts
    else:
        run_again = True

    # if force is True and needs to run again due to file checks
    if run_again or force:
        data = {}
        models = []
        if sameR:
            # create random r array
            r = GLV.get_random_r()
        for n in range(cohorts):
            if not sameR:
                # create random r array
                r = GLV.get_random_r()
            # create random A for this model
            A = GLV.get_random_A(probability=probability, bound=bound)
            model = GLV(r=r, A=A)
            # create m samples
            samples = [sample.tolist() for sample in model.get_samples(m)]
            models.append({
                "A": model.get_A(),
                "cohort": samples,
                "index": n,
                "r": r.tolist()
            })
        data['models'] = models
        # save to file
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)
    return data


def generate_random_samples(data, num):
    # create new samples - each sample is from one of the GLV models
    samples = []
    real = []
    progress = ProgressBar(num, f'creating {str(num)} samples', 25)
    for i in range(num):
        # choose cohort and generate sample with the same model
        chosen_model = random.choice(data['models'])
        model = GLV(r=chosen_model['r'], A=chosen_model['A'])
        samples.append(model.get_sample())
        real.append(chosen_model['index'])
        progress.update()
    del progress
    return samples, real


def generate_random_shuffled_samples(chosen_model, num):
    # generate random samples - part of them real and part of them are shuffles
    samples = []
    real = []
    progress = ProgressBar(num, f'creating {str(num)} shuffled samples', 25)
    for i in range(num):
        model = GLV(r=chosen_model['r'], A=chosen_model['A'])
        is_real = GLV.decision(0.5)
        samples.append(model.get_sample() if is_real else model.get_shuffled_sample(chosen_model['cohort']))
        real.append(0 if is_real else 1) # 0 means real
        progress.update()
    del progress
    return samples, real
