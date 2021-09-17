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

def set_ops_3_to_3(inputs, normalize=False):
    N, D, m, m, m = inputs.shape
    dim = inputs.shape[-1]

    # only care about marginalizing over each of the individual indices
    sum_all = inputs.sum(dim=(-1, -2, -3))
    sum_c1 = inputs.sum(dim=-1)
    sum_c2 = inputs.sum(dim=-2)
    sum_c3 = inputs.sum(dim=-3)

    sum_c12 = inputs.sum(dim=(-1, -2))
    sum_c13 = inputs.sum(dim=(-1, -3))
    sum_c23 = inputs.sum(dim=(-2, -3))

    # dont really care about diagonal
    ops = [None] * 20
    ops[1] = torch.tile(sum_all.view(N, D, 1, 1, 1), (1, 1, dim, dim, dim))

    ops[2] = torch.tile(sum_c1.unsqueeze(-1), (1, 1, 1, 1, m))
    ops[3] = torch.tile(sum_c1.unsqueeze(-2), (1, 1, 1, m, 1))
    ops[4] = torch.tile(sum_c1.unsqueeze(-3), (1, 1, m, 1, 1))

    ops[5] = torch.tile(sum_c2.unsqueeze(-1), (1, 1, 1, 1, m))
    ops[6] = torch.tile(sum_c2.unsqueeze(-2), (1, 1, 1, m, 1))
    ops[7] = torch.tile(sum_c2.unsqueeze(-3), (1, 1, m, 1, 1))

    ops[8]  = torch.tile(sum_c3.unsqueeze(-1), (1, 1, 1, 1, m))
    ops[9] = torch.tile(sum_c3.unsqueeze(-2), (1, 1, 1, m, 1))
    ops[10] = torch.tile(sum_c3.unsqueeze(-3), (1, 1, m, 1, 1))

    ops[11] = torch.tile(sum_c12.view(N, D, m, 1, 1), (1, 1, 1, m, m))
    ops[12] = torch.tile(sum_c12.view(N, D, 1, m, 1), (1, 1, m, 1, m))
    ops[13] = torch.tile(sum_c12.view(N, D, 1, 1, m), (1, 1, m, m, 1))

    ops[14] = torch.tile(sum_c13.view(N, D, m, 1, 1), (1, 1, 1, m, m))
    ops[15] = torch.tile(sum_c13.view(N, D, 1, m, 1), (1, 1, m, 1, m))
    ops[16] = torch.tile(sum_c13.view(N, D, 1, 1, m), (1, 1, m, m, 1))

    ops[17] = torch.tile(sum_c23.view(N, D, m, 1, 1), (1, 1, 1, m, m))
    ops[18] = torch.tile(sum_c23.view(N, D, 1, m, 1), (1, 1, m, 1, m))
    ops[19] = torch.tile(sum_c23.view(N, D, 1, 1, m), (1, 1, m, m, 1))

    if normalize:
        ops[1] = torch.divide(ops[1], dim * dim * dim)
        for d in range(2, 11):
            ops[d] = torch.divide(ops[d], dim)

        for d in range(11, 20):
            ops[d] = torch.divide(ops[d], dim * dim)

    return ops[1:]


def set_ops_4_to_4(inputs):
    N, D, m, _, _, _ = inputs.shape
    sum_all = inputs.sum(dim=(-1, -2, -3, -4))
    c1, c2, c3 = [], [], []

    # collapse 1 dim
    sum_c1 = inputs.sum(dim=-1)
    sum_c2 = inputs.sum(dim=-2)
    sum_c3 = inputs.sum(dim=-3)
    sum_c4 = inputs.sum(dim=-4)
    c1s = [sum_c1, sum_c2, sum_c3, sum_c4]

    # collapse 2
    sum_c12 = inputs.sum(dim=(-1, -2))
    sum_c13 = inputs.sum(dim=(-1, -3))
    sum_c14 = inputs.sum(dim=(-1, -4))
    sum_c23 = inputs.sum(dim=(-2, -3))
    sum_c24 = inputs.sum(dim=(-2, -4))
    sum_c34 = inputs.sum(dim=(-3, -4))
    c2s = [sum_c12, sum_c13, sum_c14, sum_c23, sum_c24, sum_c34]

    # collapse 3
    sum_c123 = inputs.sum(dim=(-1, -2, -3))
    sum_c124 = inputs.sum(dim=(-1, -2, -4))
    sum_c134 = inputs.sum(dim=(-1, -3, -4))
    sum_c234 = inputs.sum(dim=(-2, -3, -4))
    c3s = [sum_c123, sum_c124, sum_c134, sum_c234]

    # broadcast
    ops = []
    ops.append(torch.tile(sum_all.view(N, D, 1, 1, 1, 1), (1, 1, m, m, m, m)))

    # broadcast collapsed 1 dims
    for c1 in c1s:
        ops.append(torch.tile(c1.view(N, D, m, m, m, 1), (1, 1, 1, 1, 1, m)))
        ops.append(torch.tile(c1.view(N, D, m, m, 1, m), (1, 1, 1, 1, m, 1)))
        ops.append(torch.tile(c1.view(N, D, m, 1, m, m), (1, 1, 1, m, 1, 1)))
        ops.append(torch.tile(c1.view(N, D, 1, m, m, m), (1, 1, m, 1, 1, 1)))

    for c2 in c2s:
        ops.append(torch.tile(c2.view(N, D, m, m, 1, 1), (1, 1, 1, 1, m, m)))
        ops.append(torch.tile(c2.view(N, D, m, 1, m, 1), (1, 1, 1, m, 1, m)))
        ops.append(torch.tile(c2.view(N, D, 1, m, m, 1), (1, 1, m, 1, 1, m)))
        ops.append(torch.tile(c2.view(N, D, m, 1, 1, m), (1, 1, 1, m, m, 1)))
        ops.append(torch.tile(c2.view(N, D, 1, m, 1, m), (1, 1, m, 1, m, 1)))
        ops.append(torch.tile(c2.view(N, D, 1, 1, m, m), (1, 1, m, m, 1, 1)))

    for c3 in c3s:
        ops.append(torch.tile(c3.view(N, D, m, 1, 1, 1), (1, 1, 1, m, m, m)))
        ops.append(torch.tile(c3.view(N, D, 1, m, 1, 1), (1, 1, m, 1, m, m)))
        ops.append(torch.tile(c3.view(N, D, 1, 1, m, 1), (1, 1, m, m, 1, m)))
        ops.append(torch.tile(c3.view(N, D, 1, 1, 1, m), (1, 1, m, m, m, 1)))

    return ops

if __name__ == '__main__':
    N = 32
    D = 16
    m = 2
    x = torch.rand(N, D, m)
    x = torch.rand(N, D, m)
    x2 = torch.rand(N, D, m, m)
    x3 = torch.rand(N, D, m, m, m)
    x4 = torch.rand(N, D, m, m, m, m)
    o = ops_1_to_1(x)
    print('1->1 okay')
    o2 = ops_1_to_2(x)
    print('1->2 okay')
    o1 = ops_2_to_1(x2)
    print('2->1 okay')
    o22 = ops_2_to_2(x2)
    print('2->2 okay')

    o33 = set_ops_3_to_3(x3)
    t33 = torch.stack(o33, dim=2)
    print(t33.shape)

    o44 = set_ops_4_to_4(x4)
    t44 = torch.stack(o44, dim=2)
    print(t44.shape)
    pdb.set_trace()
