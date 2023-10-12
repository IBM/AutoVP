import os
import random
import pickle
import torch
import torchvision
import numpy as np
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

""" Setup """
def seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True

def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def Trainable_Parameter_Size(model, fname):
    f = open(fname, "a")
    total_num = 0
    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name,", number: ", param.numel(),", dtype: ", param.dtype)
            f.write(f"{name}, number: {param.numel()}, dtype: {param.dtype}\n")
    f.close()
    return 

