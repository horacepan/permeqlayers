import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import datasets, models, transforms

class LinearFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input/weight
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output*weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output*input
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class Linear(nn.Module):
    def __init__(self, bias=False):
        super(Linear, self).__init__()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(1,1))
        #self.weight = nn.Parameter(torch.Tensor(1))
        '''
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        '''
        self.weight.data = self.weight.data.normal_(mean=0,std = 0.01)+nn.Parameter(torch.ones(1,1))
        #if bias is not None:
        #    self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, None)

class torch_sigma(nn.Module):
    def __init__(self):
        super(torch_sigma, self).__init__()
        self.f1 = 160
        self.D_K_1 = 256
        self.D_K_2 = 512
        self.S_1 = 128
        self.S_2 = 64
        self.D = 16
        self.P = 4
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.deep_kernel = nn.Sequential(
            nn.Linear(self.f1,self.D_K_1),
            nn.Tanh(),
            nn.Linear(self.D_K_1,self.D_K_2),
            nn.Tanh()
        )

        self.transform_1 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.Tanh()
        )

        self.transform_2 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.Tanh()
        )
        self.transform_3 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.Tanh()
        )
        self.transform_4 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.Tanh()
        )
        self.transform_5 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.Tanh()
        )
        self.transform_6 = nn.Sequential(
            nn.Linear(self.f1,self.f1),
            nn.Tanh()
        )

        self.sigma = Linear(bias=False)


        self.classifier = nn.Sequential(
            nn.Linear(self.f1, 1)
        )

        # Weight Initilization
        with torch.no_grad():
            self.transform_1[0].weight.data = self.transform_1[0].weight.data+nn.Parameter(torch.eye(160))
            self.transform_2[0].weight.data = self.transform_2[0].weight.data+nn.Parameter(torch.eye(160))
            self.transform_3[0].weight.data = self.transform_3[0].weight.data+nn.Parameter(torch.eye(160))
            self.transform_4[0].weight.data = self.transform_4[0].weight.data+nn.Parameter(torch.eye(160))
            self.transform_5[0].weight.data = self.transform_5[0].weight.data+nn.Parameter(torch.eye(160))
            self.transform_6[0].weight.data = self.transform_6[0].weight.data+nn.Parameter(torch.eye(160))

    def message_passing(self,x,y,z,norm,init_size):
        if init_size == 4:
            xx = torch.sum(x*x,2).unsqueeze(1).repeat(1,x.shape[1],1).transpose(1,2)
            yy = torch.sum(y*y,2).unsqueeze(1).repeat(1,x.shape[1],1)
            xy = torch.matmul(x,y.transpose(1,2))
            k = -(xx+yy-2*xy)
            k = self.sigma(k)
            k = k - torch.mean(k,2).unsqueeze(2)
            if norm==1:
                sm = nn.Softmax(2)
                k = sm(k)
            z = torch.matmul(k,z)
        else:
            xx = torch.sum(x*x,1).reshape(x.shape[0],1).repeat(1,x.shape[0]).transpose(0,1)
            yy = torch.sum(y*y,1).reshape(y.shape[0],1).repeat(1,y.shape[0])
            xy = torch.mm(x,torch.t(y))
            k =  - (xx+yy-2*xy)
            k = self.sigma(k)
            k = k - torch.mean(k,1).unsqueeze(1)
            if norm==1:
                sm = nn.Softmax(1)
                k = sm(k)
            z = torch.mm(k,z)
        return z

    def forward(self, H):
        init_size = len(H.shape)
        b_size = H.shape[0]
        if init_size == 4:
            H = H.reshape(H.shape[0]*H.shape[1],1,H.shape[3],H.shape[3])
            H = self.feature_extractor_part1(H)
            H = H.view(-1, 10 * 4 * 4)
            H = H.reshape(b_size,int(H.shape[0]/b_size),10*4*4)
        else:
            H = H.unsqueeze(1)
            H = self.feature_extractor_part1(H)
            H = H.view(-1, 10 * 4 * 4)
        # Deep kernel feature extraction
        d_k_f = self.deep_kernel(H)

        # Update our features
        H1 = self.message_passing(d_k_f,d_k_f,H,1,init_size)
        H2 = (H + H1)/2
        H2 = self.transform_1(H2)
        H3 = self.message_passing(d_k_f,d_k_f,H2,1,init_size)
        H4 = (H3 + H2)/2
        H4 = self.transform_2(H4)
        H5 = self.message_passing(d_k_f,d_k_f,H4,1,init_size)
        H6 = (H5 + H4)/2
        H6 = self.transform_3(H6)
        '''
        H7 = self.message_passing(d_k_f,d_k_f,H6,1,init_size)
        H8 = (H7 + H6)/2
        H8 = self.transform_4(H8)
        H9 = self.message_passing(d_k_f,d_k_f,H8,1,init_size)
        H10 = (H9 + H8)/2
        H10 = self.transform_5(H10)
        H11 = self.message_passing(d_k_f,d_k_f,H10,1,init_size)
        H12 = (H11 + H10)/2
        H12 = self.transform_6(H12)
        '''
        #M = H.reshape(1,len(H))
        if init_size == 4:
            M = torch.max(H6,1)[0]
        else:
            M = torch.max(H6,0)[0]
        r = self.classifier(M)
        r = torch.exp(r)
        return r #,self.sigma.weight.data.item()
