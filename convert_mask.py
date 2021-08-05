import pickle
from tqdm import tqdm
from dataloader import load
import pdb
import time
from itertools import combinations
import torch
import torch.nn as nn

def main(fn):
    '''
    Given pair of five tuples -> make sparse tensor of all 5C2 things
    to pick
    '''
    Xs, ys = load(fn)
    print('Len Xs:', len(Xs))
    res1 = []
    res_lst = {0: [], 1:[]}
    sp_tensors = {}
    MAX_CHAR = 113

    for t in [0, 1]:
        for i in tqdm(range(len(Xs))):
            team  = Xs[i, t]

            inds = combinations(team, 2)
            idx = torch.LongTensor([x for x in inds]).T
            tens = torch.sparse.LongTensor(idx, torch.ones(idx.shape[1]), size=(MAX_CHAR, MAX_CHAR))
            res_lst[t].append(tens)

        try:
            mat = torch.stack(res_lst[t])
            sp_tensors[t] = mat
        except:
            pdb.set_trace()

    combined_mat = torch.stack([sp_tensors[0], sp_tensors[1]], dim=1)
    torch.save(combined_mat, f'{fn[:-4]}_mask.pt')
    print('done saving sparse mask: {}'.format(fn))

if __name__ == '__main__':
    test_fn = 'dota2Test.pkl'
    train_fn = 'dota2Train.pkl'
    main(test_fn)
    main(train_fn)
