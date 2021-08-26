import pdb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def _parse_row(row):
    ones = []
    nones = []
    for idx, val in enumerate(row[4:]):
        if val == -1:
            nones.append(idx)
        elif val == 1:
            ones.append(idx)
    return ones, nones

def load(fn, nrows=None):
    df = pd.read_csv(fn, header=None, nrows=nrows)
    cols = ['outcome', 'cluster_id', 'mode', 'type', 'heros']
    y = df[0].values
    X = []
    for _, row in tqdm(df.iterrows()):
        ones, nones = _parse_row(row)
        X.append((ones, nones))

    return X, y

def main():
    test_fn = './dota2Test.csv'
    train_fn = './dota2Train.csv'
    fns = [test_fn, train_fn]

    for fn in fns:
        X, y = load(fn)

        data = {'X': X, 'y': y}
        with open(f'{fn[:-4]}.pkl', 'wb') as f:
            pickle.dump(data, f)
            print('done pickling: {}'.format(fn))


main()
