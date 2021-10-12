import pdb
import pickle
import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from torchvision import transforms
from torchvision.datasets import Omniglot

class StochasticOmniSetData(Dataset):
    def __init__(self, omni, epoch_len, max_set_length=10):
        self.omni = omni
        self._len = epoch_len # number of set samples in an epoch
        self.max_set_length = max_set_length
        self.probs = {n: np.ones(n) / n for n in range(1, max_set_length + 1)}
        self.labels = np.array([t[1] for t in omni._flat_character_images])
        self.unique_chars = len(set(self.labels))

    def __len__(self):
        return self._len

    def __getitem__(self, bidx):
        batch_size = len(bidx)
        set_length = np.random.randint(1, self.max_set_length)
        unique_nums = np.random.randint(1,(set_length+1), batch_size)
        vals = np.zeros((batch_size, set_length), dtype=int)

        for j in range(batch_size):
            unique_num = unique_nums[j]
            char_list = np.random.choice(self.unique_chars, unique_num, replace=False)
            how_many = 1 + np.random.multinomial(set_length - unique_num, self.probs[unique_num])
            indices = []

            for i in range(len(char_list)):
                label_eq = np.where(self.labels==char_list[i])[0]
                choice = np.random.choice(len(label_eq), how_many[i],replace=False)
                index = list(label_eq[choice])
                indices += (index)

            indices = np.array(indices)
            vals[j] = indices
            # grab the images
        vals_unrolled = vals.reshape(-1)
        imgs = torch.stack([self.omni[i][0] for i in vals_unrolled])
        imgs = imgs.view(len(bidx), -1, 105, 105)
        return imgs, unique_nums, vals_unrolled
        #train_inds.append(vals)
        #train_labels.append(np.array(label_set))
        #return train_inds, train_labels

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

if __name__ == '__main__':
    epoch_len = 3
    bs = 3
    max_set_length = 2
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206],
                                         std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    omni = Omniglot(root='./data/', transform=transform, background=True, download=True)
    pdb.set_trace()
    train_dataset = StochasticOmniSetData(omni, epoch_len, max_set_length)
    test_dataset = StochasticOmniSetData(omni, epoch_len, max_set_length)

    train_dataloader = DataLoader(dataset=train_dataset,
        sampler=BatchSampler(
            SequentialSampler(train_dataset), batch_size=bs, drop_last=False
        ),
    )
    test_dataloader = DataLoader(dataset=test_dataset,
        sampler=BatchSampler(
            SequentialSampler(test_dataset), batch_size=bs, drop_last=False
        ),
    )
    seen_vals = {}
    print('first time train loader')
    for idx, batch in enumerate(train_dataloader):
        ax, ay, az = batch
        ax = ax[0]
        ay = ay[0]
        az = az[0]
        print(ay, az)
        seen_vals.update(az.tolist())

    other = {}
    print('rerunning train loader')
    for idx, batch in enumerate(train_dataloader):
        ax, ay, az = batch
        ax = ax[0]
        ay = ay[0]
        az = az[0]
        print(ay, az)
        other.update(az.tolist())

    lst = {}
    print('running test loader')
    for batch in test_dataloader:
        ax, ay, az = batch
        ax = ax[0]
        ay = ay[0]
        az = az[0]
        print(ay, az)
        lst.update(az.tolist())
    pdb.set_trace()
