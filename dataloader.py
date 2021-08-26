import pdb
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

NUM_ENTITIES = 113

def load(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        X, y = data['X'], data['y']

        X = torch.LongTensor(X)
        y = torch.from_numpy(y)
        return X, y

class Dataset(torch.utils.data.Dataset):
  def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys

  def __len__(self):
        return len(self.Xs)

  def __getitem__(self, idx):
        X = self.Xs[idx]
        y = self.ys[idx]
        return X, y

class BowDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys
        self.bow_Xs = self._process_Xs(Xs)

    def _process_Xs(self, Xs):
        t1_mat = torch.zeros(len(Xs), NUM_ENTITIES)
        t2_mat = torch.zeros(len(Xs), NUM_ENTITIES)

        for k in tqdm(range(len(Xs))):
            t1_mat[k, Xs[k, 0, :]] = 1
            t2_mat[k, Xs[k, 1, :]] = 1

        return torch.stack([t1_mat, t2_mat], axis=1)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.bow_Xs[idx], self.ys[idx]

class DataWithMask(Dataset):
    def __init__(self, Xs, X_masks, ys):
        self.Xs = Xs
        self.ys = ys
        self.X_masks = X_masks

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        X = self.Xs[idx]
        y = self.ys[idx]
        X_mask = self.X_masks[idx]
        return X, X_mask, y

def main():
    fn = './data/dota2/dota2Train.pkl'
    X, y = load(fn)
    data = Dataset(X, y)
    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    bow_data = BowDataset(X, y)
    pdb.set_trace()
    #for xbatch, ybatch in dataloader:
    #    pdb.set_trace()

if __name__ == '__main__':
    main()
