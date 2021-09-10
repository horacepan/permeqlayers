import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from equivariant_layers import ops_2_to_1, ops_1_to_2, ops_1_to_1, ops_2_to_2, set_ops_3_to_3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Eq2to1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq2to1, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.rand(in_dim, out_dim, self.basis_dim))
        self.bias = nn.Parameter(torch.rand(1, out_dim, 1))

    def forward(self, inputs):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        ops = ops_2_to_1(inputs)
        ops = torch.stack(ops, dim=2)
        output = torch.einsum('dsb,ndbi->nsi', self.coefs, ops)
        output = output + self.bias
        return output

class Eq1to2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq1to2, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim)), (in_dim, out_dim, self.basis_dim)))
        # diag bias, all bias, mat diag bias
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))

    def forward(self, inputs):
        ops = ops_1_to_2(inputs)
        ops = torch.stack(ops, dim=2)
        output = torch.einsum('dsb,ndbij->nsij', self.coefs, ops)
        output = output + self.bias
        return output

class Eq2to2(nn.Module):
    def __init__(self, in_dim, out_dim, n=23, normalize=True):
        super(Eq2to2, self).__init__()
        self.basis_dim = 15
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2./ (in_dim * out_dim * self.basis_dim)), (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))
        self.diag_bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))
        self.diag_eye = torch.eye(n).unsqueeze(0).unsqueeze(0).to(device)
        # mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0), diag_bias)

        '''
        diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        all_bias = tf.get_variable('all_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0), diag_bias)
        '''

    def forward(self, inputs):
        ops = ops_2_to_2(inputs)
        ops = torch.stack(ops, dim=2)
        output = torch.einsum('dsb,ndbij->nsij', self.coefs, ops)
        diag_bias = self.diag_eye.multiply(self.diag_bias)
        output = output + self.bias + diag_bias
        return output

class Eq1to2Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_dims):
        self.layers = nn.ModuleList(
            [Eq1to2(in_dim, out_dim)] + \
            [Eq2to2(out_dim, out_dim) for _ in range(1, len(layer_dims))]
        )
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, inputs):
        '''
        inputs:  N x D x m
        outputs: N x D x m x m
        '''
        output = inputs
        for l in self.layers:
            inputs = F.relu(l(outputs))
        output = self.fc_out(output)
        return output

class Net2to2(nn.Module):
    def __init__(self, layers, out_dim, n, out_model='Linear', **kwargs):
        '''
        layers: list of tuples (dim_in, dim_out)
        out_dim: output dimension
        n: size of input n \times n  tensor
        '''
        super(Net2to2, self).__init__()
        self.layers = nn.ModuleList([Eq2to2(din, dout, n) for din, dout in layers])
        if out_model == 'Linear':
            self.out_net = nn.Linear(layers[-1][-1], out_dim)
        elif out_model=='MLP':
            self.out_net = MLP(layers[-1][0], kwargs['mlp_hid_dim'], out_dim)

    def forward(self, x):
        '''
        x: N x d x m x m
        Returns: N x m x m x out_dim
        '''
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.permute(0, 2, 3, 1)
        output = self.out_net(x)
        return output


class SetEq3to3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SetEq3to3, self).__init__()
        self.basis_dim = 19
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2.0 / (in_dim + out_dim)),
                                  (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1, 1))

    def forward(self, x):
        ops = torch.stack(set_ops_3_to_3(x), dim=2)
        print(ops.shape)
        output = torch.einsum('dsb,ndbijk->nsijk', self.coefs, ops) # in/out/basis, batch/in/basis/ijk
        output = output + self.bias
        return output

class SetNet3to3(nn.Module):
    def __init__(self, layers, out_dim, out_model='Linear'):
        super(Net3to3, self).__init__()
        self.layers = nn.ModuleList([SetEq3to3(din, dout) for din, dout in layers])
        if out_model == 'Linear':
            self.out_net = nn.Linear(layers[-1][1], out_dim)
        else:
            self.out_net = MLP(layers[-1][1], out_dim)

    def forward(self, x):
        '''
        x: tensor of size Batch x feature x n x n x n
        Return: tensor of size Batch x n x n x n
        '''
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.permute(0, 2, 3, 4, 1)
        output = self.out_net(x)
        return output

if __name__ == '__main__':
    N = 10
    d_in = 5
    d_hid = 3
    d_out = 1
    m = 7
    x3 = torch.rand(N, d_in, m, m, m)
    x2 = torch.rand(N, d_in, m, m)
    x1 = torch.rand(N, d_in, m)

    m12a = Eq1to2(d_in, d_hid)
    m12b = Eq1to2(d_hid, d_out)

    m21a = Eq2to1(d_in, d_hid)
    m21b = Eq2to1(d_hid, d_out)

    out_dim = 4
    layers = [(d_in, 6), (6, 3), (3, out_dim)]
    net = Net2to2(layers, out_dim, m)
    print(m12b(m21a(x2)).shape, f'expect {N} x 1 x {m} x {m}')
    print(m21b(m12a(x1)).shape, f'expect {N} x 1 x {m}')
    print(net(x2).shape, 'expect dim:', f'{N} x {m} x {m} x {out_dim}')

    m33 = SetEq3to3(d_in, d_hid)
    print(m33(x3).shape, f'expect {N} x {d_hid} x {m} x {m} x {m}')
