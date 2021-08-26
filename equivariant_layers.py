import pdb
import torch
import torch.nn as nn

def check_shape(x, shape):
    assert len(x.shape) == len(shape)
    for xs, s in zip(x.shape, shape):
        assert xs == s

def ops_1_to_1(inputs, normalize=False):
    dim = inputs.shape[-1]
    N, d, m = inputs.shape

    dim = inputs.shape[-1]
    sums = inputs.sum(dim=2, keepdim=True)
    op1 = inputs
    check_shape(op1, (N, d, m))

    op2 = torch.tile(sums, (1, 1, dim))
    check_shape(op2, (N, d, m))

    return [op1, op2]

def ops_1_to_2(inputs, normalize=False):
    '''
    inputs: N x D x m tensor
    '''
    N, D, m = inputs.shape
    dim = inputs.shape[-1]
    sum_all = inputs.sum(dim=2, keepdim=True) # N x D x 1

    op1 = torch.diag_embed(inputs)
    op2 = torch.diag_embed(torch.tile(sum_all, (1, 1, dim)))
    op3 = torch.tile(inputs.unsqueeze(2), (1, 1, dim, 1))
    op4 = torch.tile(inputs.unsqueeze(3), (1, 1, 1, dim))
    op5 = torch.tile(sum_all.unsqueeze(3), (1, 1, dim, dim))
    return [op1, op2, op3, op4, op5]

def ops_2_to_1(inputs, normalize=False):
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    check_shape(diag_part, (N, D, m))

    sum_diag_part = diag_part.sum(dim=2, keepdims=True)
    check_shape(sum_diag_part, (N, D, 1))

    sum_rows = inputs.sum(dim=3)
    check_shape(sum_rows, (N, D, m))

    sum_cols = inputs.sum(dim=2)
    check_shape(sum_cols, (N, D, m))

    sum_all = inputs.sum(dim=(2,3))
    check_shape(sum_all, (N, D))

    op1 = diag_part
    op2 = torch.tile(sum_diag_part, (1, 1, dim))
    op3 = sum_rows
    op4 = sum_cols
    op5 = torch.tile(sum_all.unsqueeze(2), (1, 1, dim))
    ops = [op1, op2, op3, op4, op5]
    for idx, op in enumerate(ops):
        check_shape(op, (N, D, m))
    return [op1, op2, op3, op4, op5]

def ops_2_to_2(inputs, normalize=False):
    N, D, m, m = inputs.shape
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1) # N x D x m
    sum_diag_part = diag_part.sum(dim=2, keepdims=True) # N x D x 1
    sum_rows = inputs.sum(dim=3) # N x D x m
    sum_cols = inputs.sum(dim=2) # N x D x m
    sum_all = inputs.sum(dim=(2,3)) # N x D

    ops = [None] * (15 + 1)
    ops[1]  = torch.diag_embed(diag_part) # N x D x m x m
    ops[2]  = torch.diag_embed(torch.tile(sum_diag_part, (1, 1, dim)))
    ops[3]  = torch.diag_embed(sum_rows)
    ops[4]  = torch.diag_embed(sum_rows)
    ops[5]  = torch.diag_embed(torch.tile(sum_all.unsqueeze(-1), (1, 1, dim)))

    ops[6]  = torch.tile(sum_cols.unsqueeze(3), (1, 1, 1, dim))
    ops[7]  = torch.tile(sum_rows.unsqueeze(3), (1, 1, 1, dim))
    ops[8]  = torch.tile(sum_cols.unsqueeze(2), (1, 1, dim, 1))
    ops[9]  = torch.tile(sum_rows.unsqueeze(2), (1, 1, dim, 1))
    ops[10] = inputs
    ops[11] = torch.transpose(inputs, 2, 3)
    ops[12] = torch.tile(diag_part.unsqueeze(3), (1, 1, 1, dim))
    ops[13] = torch.tile(diag_part.unsqueeze(2), (1, 1, dim, 1))
    ops[14] = torch.tile(sum_diag_part.unsqueeze(3), (1, 1, dim, dim))
    ops[15] = torch.tile(sum_all.unsqueeze(-1).unsqueeze(-1), (1, 1, dim, dim))

    for i in range(1, 16):
        op = ops[i]
        if i == 5 or i == 15:
            ops[i] = torch.divide(op, dim * dim)
        else:
            ops[i] = torch.divide(op, dim)
    return ops[1:]

if __name__ == '__main__':
    N = 32
    D = 16
    m = 7
    x = torch.rand(N, D, m)
    x = torch.rand(N, D, m)
    x2 = torch.rand(N, D, m, m)
    o = ops_1_to_1(x)
    print('1->1 okay')
    o2 = ops_1_to_2(x)
    print('1->2 okay')
    o1 = ops_2_to_1(x2)
    print('2->1 okay')
    o22 = ops_2_to_2(x2)
    print('2->2 okay')
