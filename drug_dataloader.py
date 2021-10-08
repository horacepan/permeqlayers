import pdb
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

def _new_key(lst):
    skey = tuple(sorted(lst))
    return skey

def _load_df(fn, agg=None):
    df = pd.read_csv(fn)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.columns = ['combo', 0, 1, 2, 3]

    if agg == 'mean':
        df['value'] = df[[0, 1, 2, 3]].replace(0, np.NaN).mean(axis=1)
    elif agg == 'median':
        df['value'] = df[[0, 1, 2, 3]].replace(0, np.NaN).median(axis=1)
    return df

def _get_drug_doses(df, max_drugs=5):
    drugs = df['combo'].apply(lambda x: x.split('-'))
    dosages = drugs.apply(lambda x: tuple(int(s[-1]) for s in x))
    entities = drugs.apply(lambda x: tuple(s[:-1] for s in x))

    drugs = np.zeros((len(entities), max_drugs))
    doses = np.zeros((len(dosages), max_drugs))
    for row_id, (ent_row, dose_row) in enumerate(zip(entities, dosages)):
        drugs[row_id, :len(ent_row)] = [PrevalenceDataset.drug_cat_map[drug] for drug in ent_row]
        doses[row_id, :len(dose_row)] = [PrevalenceDataset.dose_amounts[drug][d] for drug, d in zip(ent_row, dose_row)]

    return torch.IntTensor(drugs), torch.FloatTensor(doses)

def _get_drug_categories(df, max_drugs=5):
    drugs = df['combo'].apply(lambda x: x.split('-'))
    dosages = drugs.apply(lambda x: tuple(int(s[-1]) for s in x))
    entities = drugs.apply(lambda x: tuple(s[:-1] for s in x))

    drugs = np.zeros((len(entities), max_drugs), dtype=int)
    for row_id, (ent_row, dose_row) in enumerate(zip(entities, dosages)):
        drugs[row_id, :len(ent_row)] = [3 * (PrevalenceDataset.drug_cat_map[drug] - 1) + d for drug, d in zip(ent_row, dose_row)]

    return torch.IntTensor(drugs)

def _get_bow(cats, num_ents):
    bow = np.zeros((cats.shape[0], num_ents))

    for idx, row in enumerate(cats):
        bow[idx, row] = 1

    bow[:, 0] = 0
    return torch.IntTensor(bow)

def gen_sparse_drug_data(max_drugs, train_pct, agg='median', seed=0, df_fmt='./data/prevalence/prevalence_{}.csv'):
    '''
    train data gets all rows with less than max_drugs and train_pct of max_drugs
    test data gets 1 - train_pct of max_drugs

    fn: filename of csv to load
    max_drugs: int, maximum number of drugs to consider
    Returns: tuple of train, test Dataset
    '''
    max_drug_dataset = PrevalenceDataset(df_fmt.format(max_drugs), max_drugs=max_drugs)
    train_len = int(train_pct * len(max_drug_dataset))
    test_len = len(max_drug_dataset) - train_len
    max_drug_train, max_drug_test = random_split(max_drug_dataset, (train_len, test_len),
                                                 generator=torch.Generator().manual_seed(seed))

    ds = [PrevalenceDataset(df_fmt.format(i), max_drugs=max_drugs) for i in range(1, max_drugs)] + [max_drug_train]
    train_dataset = ConcatDataset(ds)
    test_dataset = max_drug_test
    return train_dataset, test_dataset

def gen_sparse_embedded_drug_data(max_train_pct, agg='median', seed=0,
                                  df_fmt='./data/prevalence/prevalence_{}.csv'):
    max_drug_dataset = PrevalenceDataset(df_fmt.format(max_drugs), max_drugs=max_drugs)
    train_len = int(train_pct * len(max_drug_dataset))
    test_len = len(max_drug_dataset) - train_len
    max_drug_train, max_drug_test = random_split(max_drug_dataset, (train_len, test_len),
                                                 generator=torch.Generator().manual_seed(seed))

    ds = [PrevalenceDataset(df_fmt.format(i), max_drugs=max_drugs) for i in range(1, max_drugs)] + [max_drug_train]
    train_dataset = ConcatDataset(ds)
    test_dataset = max_drug_test
    return train_dataset, test_dataset

class PrevalenceDataset(torch.utils.data.Dataset):
    '''
    Prevalence dataset spits out: one hot categorical variables of the drug
    identity as well as a continous variable for the concentration
    '''
    num_entities = 8
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

    def __init__(self, fn, df=None, agg='median', max_drugs=5):
        if not df:
            df = _load_df(fn, agg=agg)
        drug_doses = _get_drug_doses(df, max_drugs)
        self.drugs = drug_doses[0]
        self.doses = drug_doses[1]
        self.ys = torch.FloatTensor(df['value'].values).reshape(-1, 1)

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

class PrevalenceTupleDataset(PrevalenceDataset):
    def __init__(self, fn, df=None, agg='median', max_drugs=5, order=3):
        super(PrevalenceTupleDataset, self).__init__(fn, df, agg, max_drugs)
        self.tup_drugs = self._gen_tups(data, order)

    def _gen_tups(self, data, order):
        self.tup_drugs = []
        xs = [x] * order
        uniques = sorted(set(prod(*xs)))
        rev_map = {tup: idx for idx, tup in enumerate(uniques)}

        for tlst in data:
            xs = prod(tlst, tlst, tlst)
            # then convert to sorted tuple of unique vals here
            # then map xs -> index of this tuple
            self.tup_drugs.append(new_key)

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        return self.tup_drugs[idx], self.doses[idx], self.ys[idx]

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
        self.ys = torch.FloatTensor(df['value'].values).reshape(-1, 1)

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        return self.drugs[idx], self.ys[idx]

class PrevalenceBowDataset(torch.utils.data.Dataset):
    num_entities = 24
    def __init__(self, fn, agg='median'):
        df = _load_df(fn, agg=agg)
        drug_cats = _get_drug_categories(df)
        self.bow = _get_bow(drug_cats, PrevalenceBowDataset.num_entities + 1)
        self.ys = torch.FloatTensor(df['value'].values).reshape(-1, 1)

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        return self.bow[idx], self.ys[idx]

if __name__ == '__main__':
    fn = './data/prevalence/parsed_deduped.csv'
    save_fn = './data/prevalence/prevalence_dataset.pkl'
    st = time.time()
    data = PrevalenceDataset(fn, agg='median')
    loader = DataLoader(data, batch_size=32)
    for batch in loader:
        pdb.set_trace()
    end = time.time()
    print('Load df time: {:.2f}s'.format(end - st))

    st = time.time()
    bow = PrevalenceBowDataset(fn)
    end = time.time()
    print('Load df time: {:.2f}s'.format(end - st))
    pdb.set_trace()
