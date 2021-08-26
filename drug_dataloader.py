import pdb
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

def _load_df(fn, agg=None):
    df = pd.read_csv(fn)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.columns = ['combo', 0, 1, 2, 3]

    if agg == 'mean':
        df['value'] = df[[0, 1, 2, 3]].replace(0, np.NaN).mean(axis=1)
    elif agg == 'median':
        df['value'] = df[[0, 1, 2, 3]].replace(0, np.NaN).median(axis=1)
    return df

def _get_drug_doses(df):
    drugs = df['combo'].apply(lambda x: x.split('-'))
    dosages = drugs.apply(lambda x: tuple(int(s[-1]) for s in x))
    entities = drugs.apply(lambda x: tuple(s[:-1] for s in x))

    drugs = np.zeros((len(entities), 5))
    doses = np.zeros((len(dosages), 5))
    for row_id, (ent_row, dose_row) in enumerate(zip(entities, dosages)):
        drugs[row_id, :len(ent_row)] = [PrevalenceDataset.drug_cat_map[drug] for drug in ent_row]
        doses[row_id, :len(dose_row)] = [PrevalenceDataset.dose_amounts[drug][d] for drug, d in zip(ent_row, dose_row)]

    return drugs, doses

def _get_drug_categories(df):
    drugs = df['combo'].apply(lambda x: x.split('-'))
    dosages = drugs.apply(lambda x: tuple(int(s[-1]) for s in x))
    entities = drugs.apply(lambda x: tuple(s[:-1] for s in x))

    drugs = np.zeros((len(entities), 5))
    for row_id, (ent_row, dose_row) in enumerate(zip(entities, dosages)):
        drugs[row_id, :len(ent_row)] = [3 * (PrevalenceDataset.drug_cat_map[drug] - 1) + d for drug, d in zip(ent_row, dose_row)]

    return drugs


class PrevalenceDataset(torch.utils.data.Dataset):
    '''
    Prevalence dataset spits out: one hot categorical variables of the drug
    identity as well as a continous variable for the concentration
    '''
    drug_cat_map = {
        'AMP': 1,
        'CPR': 2,
        'DOX': 3,
        'ERY': 4,
        'FOX': 5,
        'FUS': 6,
        'STR': 7,
        'TMP': 8
    }

    # in \muM
    dose_amounts = {
        'AMP': {1: 1.87, 2: 2.52, 3: 2.89},
        'FOX': {1: 0.78, 2: 1.37, 3: 1.78},
        'TMP': {1: 0.07, 2: 0.15, 3: 0.22},
        'CPR': {1: 0.01, 2: 0.02, 3: 0.03},
        'STR': {1: 12.25, 2: 16.6, 3: 19.04},
        'DOX': {1: 0.15, 2: 0.27, 3: 0.35},
        'ERY': {1: 1.78, 2: 8.29, 3: 16.62},
        'FUS': {1: 37.85, 2: 71.01, 3: 94.42},
    }

    def __init__(self, fn, agg='median'):
        df = _load_df(fn, agg=agg)
        drug_doses = _get_drug_doses(df)
        self.drugs = drug_doses[0]
        self.doses = drug_doses[1]
        self.ys = df['value'].values

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        return self.drugs[idx], self.doses[idx], self.ys[idx]

    def save_pkl(self, save_fn):
        data = {'drugs': self.drugs, 'doses': self.doses, 'ys': self.ys}
        with open(save_fn, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def from_pkl(cls, pkl_fn):
        self = cls.__new__(cls)
        with open(pkl_fn, 'rb') as f:
            data = pickle.load(f)
            self.drugs = data['drugs']
            self.doses = data['doses']
            self.ys = data['ys']
            return self

class PrevalenceCategoricalDataset(torch.utils.data.Dataset):
    num_entities = 24

    def __init__(self, fn, agg='median'):
        df = _load_df(fn, agg=agg)
        self.drugs = _get_drug_categories(df)
        self.ys = df['value'].values

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        return self.drugs[idx], self.ys[idx]

if __name__ == '__main__':
    fn = './data/parsed_deduped.csv'
    save_fn = './data/prevalence_dataset.pkl'
    st = time.time()
    data = PrevalenceDataset(fn, agg='median')
    end = time.time()
    print('Load df time: {:.2f}s'.format(end - st))
    other = PrevalenceCategoricalDataset(fn)
    pdb.set_trace()
