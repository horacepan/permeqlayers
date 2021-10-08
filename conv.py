import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

def _compute_hid_dim(in_dim, conv_layers):
    dim = in_dim
    for conv in conv_layers:
        if conv.stride[0] == 1:
            dim -= 2 * conv.stride[0]
        else:
            dim = dim // conv.stride[0]

        dim = dim // 2 # max pool
    return conv_layers[-1].out_channels * dim * dim

class BasicConvNet(nn.Module):
    def __init__(self, in_dim, nout, nchannels=8):
        super(BasicConvNet, self).__init__()
        self.in_dim = in_dim
        self.conv1 = nn.Conv2d(1, nchannels, kernel_size=3)
        self.conv2 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.conv3 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.conv4 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.hid_dim = _compute_hid_dim(in_dim, self.conv_layers)
        self.fc = nn.Linear(self.hid_dim, nout)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2), 2)

        x = x.view(len(x), -1)
        x = self.fc(x)
        return x

def main():
    B = 32
    nout = 32
    nin = 105
    xs = torch.rand(B, 1, nin, nin)
    net = BasicConvNet(nin, nout)
    print(net(xs).shape)

if __name__ == '__main__':
    main()
