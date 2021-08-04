import pdb
import pickle
import torch
from torch.utils.data import DataLoader

def load(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        X, y = data['X'], data['y']

        X = torch.LongTensor(X)
        y = torch.from_numpy(y)
        return X, y

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, Xs, ys):
        'Initialization'
        self.Xs = Xs
        self.ys = ys

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.Xs)

  def __getitem__(self, idx):
        'Generates one sample of data'
        # Select sample
        X = self.Xs[idx, :, :4]
        y = self.ys[idx]
        return X, y

def main():
    fn = 'dota2Test.pkl'
    X, y = load(fn)
    data = Dataset(X, y)
    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    for xbatch, ybatch in dataloader:
        pdb.set_trace()
if __name__ == '__main__':
    main()
