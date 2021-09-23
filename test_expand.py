import pdb
import time
import torch
from equivariant_layers import ops_1_to_1, ops_1_to_2, ops_2_to_1, ops_2_to_2, set_ops_3_to_3, set_ops_4_to_4
from equivariant_layers_expand import eops_1_to_1, eops_1_to_2, eops_2_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4
from utils import check_memory
from eq_models import SetEq3to3, SetEq4to4, Eq2to2
from torch.profiler import profile, record_function, ProfilerActivity
from main_drug_eq import Eq1Net, Eq2Net, Eq3Net, Eq4Net

def do_profile(ofunc, x):
    with profile(activities=[
            ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            ofunc(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    check_memory()
    B = 256
    embed_dim = 32
    hid_dim = 32
    out_dim = 32
    m2  = Eq2Net(9, embed_dim, [(embed_dim + 1, hid_dim)], out_dim, ops_func=ops_2_to_2).to(device)
    me2 = Eq2Net(9, embed_dim, [(embed_dim + 1, hid_dim)], out_dim, ops_func=eops_2_to_2).to(device)
    m3  = Eq3Net(9, embed_dim, [(embed_dim + 1, hid_dim)], out_dim, ops_func=set_ops_3_to_3).to(device)
    me3 = Eq3Net(9, embed_dim, [(embed_dim + 1, hid_dim)], out_dim, ops_func=eset_ops_3_to_3).to(device)
    m4  = Eq4Net(9, embed_dim, [(embed_dim + 1, hid_dim)], out_dim, ops_func=set_ops_4_to_4).to(device)
    me4 = Eq4Net(9, embed_dim, [(embed_dim + 1, hid_dim)], out_dim, ops_func=eset_ops_4_to_4).to(device)
    oms = [m2, m3, m4]
    ems = [ me2, me3, me4]

    x = torch.randint(0, 9, (B, 5)).to(device)
    f = torch.rand(B, 5).to(device)
    check_memory()
    i = 1

    for idx, (mo, me) in enumerate(zip(oms, ems)):
        print(i); i+=1
        st = time.time()
        r1 = mo(x, f)
        end = time.time()
        print('Elapsed: {:.5f}s'.format(end - st))
        ot = end - st

        st = time.time()
        r2 = me2(x, f)
        end = time.time()
        et = end - st
        print('Elapsed: {:.5f}s'.format(end - st))
        check_memory()
        print('Ratio: {:.4f}'.format(ot / et))

def main2():
    B = 32
    d = 32
    m = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x1 = torch.rand(B, d, m).to(device)
    x2 = torch.rand(B, d, m, m).to(device)
    x3 = torch.rand(B, d, m, m, m).to(device)
    x4 = torch.rand(B, d, m, m, m, m).to(device)
    check_memory()

    args = [x1, x1, x2, x2, x3, x4]
    ofs = [ops_1_to_1, ops_1_to_2, ops_2_to_1, ops_2_to_2, set_ops_3_to_3, set_ops_4_to_4]
    efs = [eops_1_to_1, eops_1_to_2, eops_2_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4]

    for x, of, ef in zip(args, ofs, efs):
        st = time.time()
        os.append(of(x))
        ot = time.time() - st

        st = time.time()
        es.append(ef(x))
        et = time.time() - st
    check_memory()

if __name__ == '__main__':
    main()
