import pdb
import pickle
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eq_models import Eq1to2

from dataloader import Dataset, DataWithMask
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

class Eq1to2Net(nn.Module):
    def __init__(self, embed_dim, hid_dim, num_classes=2):
        super(Eq1to2Net, self).__init__()
        self.embed = nn.Embedding(NHEROS, embed_dim)
        self.eq1to2 = Eq1to2(embed_dim, hid_dim)
        self.fc_out = nn.Linear(2 * hid_dim, num_classes)

    def forward(self, x1, x2):
        embed1 = F.relu(self.embed(x1)).permute(0, 2, 1)
        embed2 = F.relu(self.embed(x2)).permute(0, 2, 1)
        eq1_pair = F.relu(self.eq1to2(embed1))
        eq2_pair = F.relu(self.eq1to2(embed2))
        pair_embed = torch.hstack([eq1_pair.sum(axis=(-1, -2)), eq2_pair.sum(axis=(-1, -2))])
        #return self.fc_out(F.dropout(pair_embed, training=self.training))
        return self.fc_out(pair_embed)

class Eq1to2Set(nn.Module):
    def __init__(self, embed_dim, hid_dim, num_classes=2):
        super(Eq1to2Set, self).__init__()
        self.embed = nn.Embedding(NHEROS, embed_dim)
        self.eq1to2 = Eq1to2(embed_dim, hid_dim)
        self.set_embed1 = DeepSets(hid_dim, hid_dim, hid_dim)
        self.set_embed2 = DeepSets(hid_dim, hid_dim, hid_dim)
        self.fc_out = nn.Linear(2 * hid_dim, num_classes)

    def forward(self, x1, x2):
        embed1 = F.relu(self.embed(x1)).permute(0, 2, 1)
        embed2 = F.relu(self.embed(x2)).permute(0, 2, 1)
        eq1_pair = F.relu(self.eq1to2(embed1))
        eq2_pair = F.relu(self.eq1to2(embed2))

        B, d, _, _ = eq1_pair.shape
        eq1_set = F.relu(self.set_embed1(eq1_pair.reshape(B, d, -1).permute(0, 2, 1)))
        eq2_set = F.relu(self.set_embed2(eq2_pair.reshape(B, d, -1).permute(0, 2, 1)))
        team_embed = torch.hstack([eq1_set, eq2_set])
        #return self.fc_out(F.dropout(team_embed, training=self.training))
        return self.fc_out(team_embed)

def main(args):
    print(args)
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    train_fn = 'dota2Train.pkl'
    test_fn = 'dota2Test.pkl'
    print('Training on: {} | Train file: {} | Test file: {}'.format(
        device, train_fn, test_fn))

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
    if args.model == 'Eq1to2Net':
        print('Using model Eq1to2Net')
        model = Eq1to2Net(args.embed_dim, args.hid_dim).to(device)
    elif args.model == 'Eq1to2Set':
        print('Using model Eq1to2Set')
        model = Eq1to2Set(args.embed_dim, args.hid_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                #model.eval()
                for xtb, ytb in test_dataloader:
                    xtb, ytb = xtb.to(device), ytb.to(device)
                    ytp = model(xtb[:, 0], xtb[:, 1])
                    _, predicted = torch.max(ytp.data, 1)
                    correct += (predicted == ytb).sum().item()
                acc = correct / len(test_data)
                coef_max = model.eq1to2.coefs.data.max().item()
                coef_min = model.eq1to2.coefs.data.min().item()
                coef_std = model.eq1to2.coefs.data.std().item()
                print('Epoch: {:4d} | Num updates: {:5d} | Running Loss: {:.3f} | Running acc: {:.3f} | Test Acc: {:.3f} | Time: {:.2f}mins | Max: {:.3f} | Min: {:.3f} | std: {:.3f} | weight decay'.format(
                    e, nupdates, running_loss, running_acc, acc, (time.time() - st) / 60.,
                    coef_max, coef_min, coef_std
                ))
                running_loss = 0
                running_acc = 0
                #model.train()
            nupdates += 1

    print('hdim: {:3d} | edim: {:3d} | model: {} | final test acc: {:.3f}'.format(args.hid_dim, args.embed_dim, args.model, acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model', type=str, default='Eq1to2Net')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--print_update', type=int, default=1000)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
