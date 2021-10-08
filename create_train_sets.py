import time
import pdb
from tqdm import tqdm
import os
import torch
import numpy as np
import pickle
from torchvision.datasets import Omniglot

def make_dataset(labels, batch_size, total_batches, unique_chars):
    train_labels = []
    train_inds = []
    probs = {v: np.ones(v) / v for v in range(1, 11)}
    set_lens = [6, 7, 8, 9, 10]

    for j in tqdm(range(total_batches)):
        label_set = []
        batch_indices = []
        count = 0
        set_length = set_lens[j % len(set_lens)]
        unique_nums = np.random.randint(1,(set_length+1), batch_size)
        vals = np.zeros((batch_size, set_length), dtype=int)
        for j in range(batch_size):
            unique_num = unique_nums[j]
            char_list = np.random.choice(unique_chars, unique_num, replace=False)
            how_many = 1 + np.random.multinomial(set_length - unique_num, probs[unique_num])
            indices = []

            for i in range(len(char_list)):
                label_eq = np.where(labels==char_list[i])[0]
                choice = np.random.choice(len(label_eq), how_many[i],replace=False)
                index = list(label_eq[choice])
                indices += (index)

            indices = np.array(indices)
            batch_indices.append(indices)
            label_set.append(unique_num)
            vals[j] = indices

        train_inds.append(vals)
        train_labels.append(np.array(label_set))
    return train_inds, train_labels

def main(data, labels, dir_pref, seed):
    batch_size = 32
    train_batches = 2000
    test_batches = train_batches // 5
    unique_chars = len(set(labels))
    st = time.time()
    train_inds, train_targets = make_dataset(labels, batch_size, train_batches, unique_chars)
    print('Done with train batches: {:.2f}s'.format(time.time() - st))

    st = time.time()
    test_inds, test_targets = make_dataset(labels, batch_size, test_batches, unique_chars)
    print('Done with test batches: {:.2f}s'.format(time.time() - st))

    with open(f'{dir_pref}/train_idx_{seed}.pkl','wb') as tr_idx:
        pickle.dump(train_inds, tr_idx)

    with open(f'{dir_pref}/train_targets_{seed}.pkl','wb') as tr_tgt:
        pickle.dump(train_targets, tr_tgt)

    with open(f'{dir_pref}/test_idx_{seed}.pkl', 'wb') as te_idx:
        pickle.dump(test_inds, te_idx)

    with open(f'{dir_pref}/test_targets_{seed}.pkl', 'wb') as te_tgt:
        pickle.dump(test_targets, te_tgt)

    print('Done')

if __name__ == '__main__':
    data = Omniglot(root='./data/', background=True, download=True)
    labels = np.array([t[1] for t in data._flat_character_images])
    dir_pref = './data/omniglot-py/unique_chars_dataset'
    for s in range(1, 6):
        np.random.RandomState(s)
        main(data, labels, dir_pref, s)
