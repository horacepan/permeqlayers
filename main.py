import pdb
import pickle
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import Dataset
from models import DeepSets

NHEROS = 113

def ncorrect(output, tgt):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == tgt).sum().item()
    return correct

def load(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        X, y = data['X'], data['y']

        y[y == -1] = 0
        X = torch.LongTensor(X)
        y = torch.from_numpy(y)
        return X, y

class BaselineEmbedDeepSets(nn.Module):
    def __init__(self, nembed, embed_dim, hid_dim, num_classes=2):
        super(BaselineEmbedDeepSets, self).__init__()
        self.embed = nn.Embedding(nembed, embed_dim)
        self.set_embed1 = DeepSets(embed_dim, hid_dim, hid_dim)
        self.set_embed2 = DeepSets(embed_dim, hid_dim, hid_dim)
        self.fc_out = nn.Linear(2 * hid_dim, num_classes)

    def forward(self, x1, x2):
        embed1 = F.relu(self.embed(x1))
        embed2 = F.relu(self.embed(x2))
        set1 = F.relu(self.set_embed1(embed1))
        set2 = F.relu(self.set_embed1(embed2))
        sets = torch.hstack([set1, set2])
        return self.fc_out(sets)

def main(args):
    print(args)
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    test_fn = './data/dota2/dota2Test.pkl'
    train_fn = './data/dota2/dota2Train.pkl'

    Xtrain, ytrain = load(train_fn)
    Xtest, ytest = load(test_fn)

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
    }
    train_data = Dataset(Xtrain, ytrain)
    test_data = Dataset(Xtest, ytest)
    train_dataloader = DataLoader(train_data, **params)
    test_dataloader = DataLoader(test_data, **params)

    loss_func = nn.CrossEntropyLoss()
    model = BaselineEmbedDeepSets(NHEROS, args.embed_dim, args.hid_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    nupdates = 0
    running_loss = 0
    running_acc = 0
    st = time.time()

    for e in range(args.epochs + 1):
        for xbatch, ybatch in train_dataloader:
            opt.zero_grad()
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            ypred = model(xbatch[:, 0], xbatch[:, 1])
            loss = loss_func(ypred, ybatch)
            batch_acc = ncorrect(ypred, ybatch) / len(ypred)
            running_loss += loss.item() / args.print_update
            running_acc += batch_acc / args.print_update
            loss.backward()
            opt.step()

            if nupdates % args.print_update == 0:
                correct = 0
                for xtb, ytb in test_dataloader:
                    xtb, ytb = xtb.to(device), ytb.to(device)
                    ytp = model(xtb[:, 0], xtb[:, 1])
                    _, predicted = torch.max(ytp.data, 1)
                    correct += (predicted == ytb).sum().item()
                acc = correct / len(test_data)

                print('Epoch: {:4d} | Num updates: {:5d} | Running Loss: {:.3f} | Running acc: {:.3f} | Test Acc: {:.3f} | Time: {:.2f}mins'.format(
                    e, nupdates, running_loss, running_acc, acc, (time.time() - st) / 60.
                ))
                running_loss = 0
                running_acc = 0
            nupdates += 1
    print('hdim: {:3d} | edim: {:3d} | final test acc: {:.3f}'.format(args.hid_dim, args.embed_dim, acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
