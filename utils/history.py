import numpy as np
import pickle

import matplotlib.pyplot as plt


def save_histories(lhist, path):
    with open(path, 'wb') as f:
        pickle.dump(lhist,f)

def load_histories(path):
    with open(path, 'rb') as f:
        lhist = pickle.load(f)
    return lhist

def lhist_to_dictarr(lhist):
    dictarr = {}
    l = []
    keys = lhist[0].keys()
    for key in keys:
        for hist in lhist:
            l.append(hist[key])
        dictarr[key] = np.array(l)
        l = []
    return dictarr