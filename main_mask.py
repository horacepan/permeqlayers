import pdb
import pickle
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eq_models import Eq1to2, Eq2to2
from main import BaselineEmbedDeepSets

from dataloader import Dataset, DataWithMask, BowDataset
from models import DeepSets

NHEROS = 113
TEAM_SIZE = 5

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
        self.eq1to2_1 = Eq1to2(embed_dim, hid_dim)
        self.eq1to2_2 = Eq1to2(embed_dim, hid_dim)
        self.set_embed1 = DeepSets(hid_dim, hid_dim, hid_dim)
        self.set_embed2 = DeepSets(hid_dim, hid_dim, hid_dim)
        self.fc_out = nn.Linear(2 * hid_dim, num_classes)

    def forward(self, x1, x2):
        embed1 = F.relu(self.embed(x1)).permute(0, 2, 1)
        embed2 = F.relu(self.embed(x2)).permute(0, 2, 1)
        eq1_pair = F.relu(self.eq1to2_1(embed1))
        eq2_pair = F.relu(self.eq1to2_2(embed2))

        B, d, _, _ = eq1_pair.shape
        eq1_set = F.relu(self.set_embed1(eq1_pair.reshape(B, d, -1).permute(0, 2, 1)))
        eq2_set = F.relu(self.set_embed2(eq2_pair.reshape(B, d, -1).permute(0, 2, 1)))
        team_embed = torch.hstack([eq1_set, eq2_set])
        return self.fc_out(F.dropout(team_embed, training=self.training))
        #return self.fc_out(team_embed)

class Eq1to2Combo(nn.Module):
    def __init__(self, embed_dim, hid_dim, num_classes=2):
        super(Eq1to2Combo, self).__init__()
        self.embed = nn.Embedding(NHEROS, embed_dim)
        # linear -> outer -> 2 -> 2 -> sum - > cat -> linear
        self.e1 = Eq2to2(embed_dim, hid_dim, n=TEAM_SIZE)
        self.e2 = Eq2to2(embed_dim, hid_dim, n=TEAM_SIZE)
        self.e3 = Eq2to2(embed_dim, hid_dim, n=TEAM_SIZE)
        self.fc_out = nn.Linear(3 * hid_dim, num_classes)
        #self.fc_out = nn.Linear(3 * hid_dim, num_classes)

    def forward(self, x1, x2):
        embed1 = F.relu(self.embed(x1)).permute(0, 2, 1)
        embed2 = F.relu(self.embed(x2)).permute(0, 2, 1)

        # 2->2
        prod1 =     torch.einsum('bdi, bdj->bdij', embed1, embed1)
        prod2 =     torch.einsum('bdi, bdj->bdij', embed2, embed2)
        pair_prod = torch.einsum('bdi, bdj->bdij', embed1, embed2)
        catted = torch.hstack([
                    F.relu(self.e1(prod1)).sum(axis=(-1, -2)),
                    F.relu(self.e2(prod2)).sum(axis=(-1, -2)),
                    F.relu(self.e3(pair_prod)).sum(axis=(-1, -2))
                 ])
        #output = self.fc_out(catted)
        output = self.fc_out(F.dropout(catted, training=self.training))
        return output

class Eq1to2Combo2(nn.Module):
    def __init__(self, embed_dim, hid_dim, num_classes=2):
        super(Eq1to2Combo2, self).__init__()
        self.embed = nn.Embedding(NHEROS, embed_dim)
        # linear -> outer -> 2 -> 2 -> sum - > cat -> linear
        self.e1 = Eq2to2(embed_dim, hid_dim, n=TEAM_SIZE)
        self.e2 = Eq2to2(embed_dim, hid_dim, n=TEAM_SIZE)
        self.e3 = Eq2to2(embed_dim, hid_dim, n=TEAM_SIZE)
        self.eq1to2_1 = Eq1to2(embed_dim, hid_dim)
        self.eq1to2_2 = Eq1to2(embed_dim, hid_dim)
        self.fc_out = nn.Linear(5 * hid_dim, num_classes)
        #self.fc_out = nn.Linear(3 * hid_dim, num_classes)

    def forward(self, x1, x2):
        embed1 = F.relu(self.embed(x1)).permute(0, 2, 1)
        embed2 = F.relu(self.embed(x2)).permute(0, 2, 1)

        # 2->2
        eq1_pair = F.relu(self.eq1to2_1(embed1))
        eq2_pair = F.relu(self.eq1to2_2(embed2))
        pair_embed = torch.hstack([eq1_pair.sum(axis=(-1, -2)), eq2_pair.sum(axis=(-1, -2))])

        prod1 = torch.einsum('bdi, bdj->bdij', embed1, embed1)
        prod2 = torch.einsum('bdi, bdj->bdij', embed2, embed2)
        pair_prod = torch.einsum('bdi, bdj->bdij', embed1, embed2)
        catted = torch.hstack([
                    eq1_pair.sum(axis=(-1, -2)),
                    eq2_pair.sum(axis=(-1, -2)),
                    F.relu(self.e1(prod1)).sum(axis=(-1, -2)),
                    F.relu(self.e2(prod2)).sum(axis=(-1, -2)),
                    F.relu(self.e3(pair_prod)).sum(axis=(-1, -2))
                 ])
        #output = self.fc_out(catted)
        output = self.fc_out(F.dropout(catted, training=self.training))
        return output
class LogisticRegression(nn.Module):
    def __init__(self, dim, num_classes=2):
        super(LogisticRegression, self).__init__()
        self.w = nn.Linear(dim, num_classes)

    def forward(self, x):
        #return F.log_softmax(self.w(x))
        return (self.w(x))

def main(args):
    print(args)
    torch.random.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    train_fn = './data/dota2/dota2Train.pkl'
    test_fn = './data/dota2/dota2Test.pkl'
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
    elif args.model == 'Eq1to2Combo':
        print('Using model Eq1to2Combo')
        model = Eq1to2Combo(args.embed_dim, args.hid_dim).to(device)
    elif args.model == 'Eq1to2Combo2':
        print('Using model Eq1to2Combo2')
        model = Eq1to2Combo2(args.embed_dim, args.hid_dim).to(device)
    elif args.model == 'BaselineEmbedDeepSets':
        print('Using model Baseline')
        model = BaselineEmbedDeepSets(NHEROS, args.embed_dim, args.hid_dim).to(device)
    elif args.model == 'LogisticRegression':
        print('Using log regr Baseline')
        model = LogisticRegression(2 * NHEROS)
        train_data = BowDataset(train_data.Xs, train_data.ys)
        test_data = BowDataset(train_data.Xs, train_data.ys)
        train_dataloader = DataLoader(train_data, **params)
        test_dataloader = DataLoader(test_data, **params)
        #loss_func = nn.NLLLoss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nupdates = 0
    running_loss = 0
    running_acc = 0
    st = time.time()

    for e in range(args.epochs + 1):
        for xbatch, ybatch in train_dataloader:
            opt.zero_grad()
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            if args.model != 'LogisticRegression':
                ypred = model(xbatch[:, 0], xbatch[:, 1])
            else:
                ypred = model(xbatch.reshape(len(xbatch), -1))
            loss = loss_func(ypred, ybatch)
            batch_acc = ncorrect(ypred, ybatch) / len(ypred)
            running_loss += loss.item() / args.print_update
            running_acc += batch_acc / args.print_update
            loss.backward()
            opt.step()

            if nupdates % args.print_update == 0:
                correct = 0
                model.eval()
                for xtb, ytb in test_dataloader:
                    xtb, ytb = xtb.to(device), ytb.to(device)
                    if args.model != 'LogisticRegression':
                        ytp = model(xtb[:, 0], xtb[:, 1])
                    else:
                        ytp = model(xtb.reshape(len(xtb), -1))
                    _, predicted = torch.max(ytp.data, 1)
                    correct += (predicted == ytb).sum().item()
                acc = correct / len(test_data)
                if hasattr(model, 'eq1to2_1'):
                    coef_max = model.eq1to2_1.coefs.data.max().item()
                    coef_min = model.eq1to2_1.coefs.data.min().item()
                    coef_std = model.eq1to2_1.coefs.data.std().item()
                else:
                    coef_max = 0
                    coef_min = 0
                    coef_std = 0
                print('Epoch: {:4d} | Num updates: {:5d} | Running Loss: {:.3f} | Running acc: {:.3f} | Test Acc: {:.3f} | Time: {:.2f}mins | Max: {:.3f} | Min: {:.3f} | std: {:.3f} | two eq1-2 dropout wd'.format(
                    e, nupdates, running_loss, running_acc, acc, (time.time() - st) / 60.,
                    coef_max, coef_min, coef_std
                ))
                running_loss = 0
                running_acc = 0
                model.train()
            nupdates += 1

    print('hdim: {:3d} | edim: {:3d} | model: {} | final test acc: {:.3f}'.format(args.hid_dim, args.embed_dim, args.model, acc))
    pdb.set_trace()

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
