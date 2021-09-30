import pdb
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import Dataset
from models import DeepSets
from dota2_models import BaselineEmbedDeepSets, Dota2Eq2Embed, Dota2Eq3Embed

NHEROS = 113

def reset_params(model):
    for p in model.parameters():
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)

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

def model_dispatch(args, device):
    if args.model == 'baseline':
        model = BaselineEmbedDeepSets(NHEROS, args.embed_dim, args.hid_dim)
    elif args.model == 'eq2':
        model = Dota2Eq2Embed(NHEROS, args.embed_dim, args.hid_dim, args.hid_dim)
    elif args.model == 'eq3':
        model = Dota2Eq3Embed(NHEROS, args.embed_dim, args.hid_dim, args.hid_dim)

    for p in model.parameters():
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)
    return model.to(device)

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
        'pin_memory': True,
    }
    train_data = Dataset(Xtrain, ytrain)
    test_data = Dataset(Xtest, ytest)
    train_dataloader = DataLoader(train_data, **params)
    test_dataloader = DataLoader(test_data, **params)

    loss_func = nn.CrossEntropyLoss()
    model = model_dispatch(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    nupdates = 0
    st = time.time()

    for e in range(args.epochs + 1):
        ep_loss = []
        ep_acc = []
        for xbatch, ybatch in train_dataloader:
            opt.zero_grad()
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            ypred = model(xbatch)
            loss = loss_func(ypred, ybatch)
            batch_acc = ncorrect(ypred, ybatch) / len(ypred)
            ep_loss.append(loss.item())
            ep_acc.append(batch_acc)
            loss.backward()
            opt.step()

        if e % args.print_update == 0:
            correct = 0
            for xtb, ytb in test_dataloader:
                xtb, ytb = xtb.to(device), ytb.to(device)
                ytp = model(xtb)
                _, predicted = torch.max(ytp.data, 1)
                correct += (predicted == ytb).sum().item()
            acc = correct / len(test_data)

            print('Epoch: {:4d} | Last ep Loss: {:.3f}, acc: {:.3f} | Test Acc: {:.3f} | Time: {:.2f}mins'.format(
                e, np.mean(ep_loss), np.mean(ep_acc), acc, (time.time() - st) / 60.
            ))
    print('Done training with input args:')
    print(args)
    print('final test acc: {:.3f}'.format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
