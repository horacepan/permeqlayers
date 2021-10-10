import pdb
import pickle
import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset

class OmniSetData(Dataset):
    def __init__(self, data, targets, omni):
        '''
        data: dict from seq len -> tensor
        target: dict from seq len -> tensor
        imgs: numpy array of images
        '''
        self.data = data
        self.targets = targets
        self.omni = omni
        self._len = sum([len(x) for x in data.values()])
        self._min_set_len = min(data.keys())
        self._lens = sorted(data.keys())
        self._total_each = data[self._min_set_len].shape[0]
        self._img_h = omni[0][0].shape[1]
        self._img_w = omni[0][0].shape[2]

    @staticmethod
    def from_files(idx_pkl, tgt_pkl, omni, fraction=1):
        xmaps = {}
        ymaps = {}
        xs = pickle.load(open(idx_pkl, 'rb'))
        ys = pickle.load(open(tgt_pkl, 'rb'))

        for x, y in zip(xs, ys):
            n = x.shape[1]
            if n not in xmaps:
                xmaps[n] = []
                ymaps[n] = []
            frac_len = int(len(x) * fraction)
            xmaps[n].append(x[:frac_len])
            ymaps[n].append(y[:frac_len])

        flattened_xs = {n: np.vstack(xmaps[n]) for n in xmaps.keys()}
        flattened_ys = {n: np.concatenate(ymaps[n]) for n in ymaps.keys()}
        return OmniSetData(flattened_xs, flattened_ys, omni)

    def __len__(self):
        return self._len

    def __getitem__(self, bidx):
        seq_len = self._min_set_len + (bidx[0] % len(self._lens))
        new_bidx = [b % self._total_each for b in bidx]
        idxs = self.data[seq_len][new_bidx]
        idxs_unrolled = idxs.reshape(-1)
        bimgs = torch.stack([self.omni[i][0] for i in idxs_unrolled])
        bimgs = bimgs.reshape(len(new_bidx), -1, self._img_h, self._img_w)
        return bimgs, self.targets[seq_len][new_bidx]

# DEPRECATED
class SetDataset(Dataset):
    def __init__(self, dataset, set_size):
        self.dataset = dataset
        self.set_size = set_size
        self.samples = torch.randint(0, len(self.dataset),
                                     size=(len(self.dataset), set_size))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vals = [self.dataset[x] for x in sample]
        ys = torch.tensor([v[1] for v in vals])
        xs = [v[0] for v in vals]
        xs = torch.stack(xs)
        nuniques = len(set(vals))
        return xs, nuniques

    def __len__(self):
        return len(self.dataset)
