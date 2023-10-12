import auto_vp.datasets as datasets
from auto_vp.wrapper import BaseWrapper
from auto_vp.utilities import setup_device
from auto_vp.dataprepare import DataPrepare, Data_Scalability
from auto_vp import programs
from auto_vp.training_process import *
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD

import argparse
import torchvision
import numpy as np
import torch
import random
import torchvision.models as models
import clip
import timm
from torchvision import transforms

import time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset', choices=["CIFAR10", "CIFAR10-C", "CIFAR100", "Melanoma", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "UCF101", "FMoW"], required=True)
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--download', type=int, choices=[0, 1], default=0)
    p.add_argument(
        '--pretrained', choices=["vgg16_bn", "resnet18", "resnet50", "resnext101_32x8d", "ig_resnext101_32x8d", "vit_b_16", "clip", "clip_large", "swin_t"], default="clip")
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001) # IG: 0.001, SWIN: 0.001 or 0.0001
    p.add_argument('--seed', type=int, default=7)

    p.add_argument('--scalibility_rio', type=int, choices=[1, 2, 4, 10, 100], default=1)
    p.add_argument('--scalibility_mode', choices=["equal", "random"], default="equal") 
    p.add_argument('--baseline', choices=["LP", "FF", "Scartch", "CLIP_TP", "CLIP_LP"], default="CLIP_LP") 
    args = p.parse_args()

    start_time = time.time()
    # set random seed
    set_seed(args.seed)

    # device setting
    device, list_ids = setup_device(1)
    print("device: ", list_ids)

    # Dataset Setting 
    channel = 3
    img_resize = 224 
    class_num = CLASS_NUMBER[args.dataset]
    random_state = args.seed   

    # choice: [True, False]. Download the dataset or not.
    if(args.download  > 0):
        download  = True
    else:
        download  = False
    
    # Initialize Model
    if args.pretrained == "vgg16_bn": # VGG-16 with batch normalization
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT) 
    elif args.pretrained == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.pretrained == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.pretrained == "resnext101_32x8d":
        model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
    elif args.pretrained  == "ig_resnext101_32x8d": # https://paperswithcode.com/model/ig-resnext?variant=ig-resnext101-32x8d
        model = timm.create_model('ig_resnext101_32x8d', pretrained=True)
    elif args.pretrained == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    elif args.pretrained == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    elif args.pretrained == "clip":
        model, clip_preprocess = clip.load("ViT-B/32", device=device)
        # https://github.com/openai/CLIP/issues/57
        for p in model.parameters(): 
            p.data = p.data.float() 
            if p.grad:
                p.grad.data = p.grad.data.float()
    elif args.pretrained == "clip_large":
        model, clip_preprocess = clip.load("ViT-L/14", device=device)
        for p in model.parameters(): 
            p.data = p.data.float() 
            if p.grad:
                p.grad.data = p.grad.data.float()

    else:
        raise NotImplementedError(f"{args.pretrained} not supported")
    
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(NETMEAN[args.pretrained], NETSTD[args.pretrained]),
    ])
    

    if(args.pretrained[0:4] == "clip"):
        clip_transform = clip_preprocess
    else:
        clip_transform = preprocess

    # dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath, target_size=(
        img_resize, img_resize), mean=NETMEAN[args.pretrained], std=NETSTD[args.pretrained], download=download, batch_size=BATCH_SIZE[args.dataset], random_state=random_state, clip_transform=clip_transform)
    
    wild_ds_list = ["Camelyon17", "Iwildcam", "FMoW"]
    if args.dataset in wild_ds_list:
        wild_dataset = True
    else:
        wild_dataset = False

    if(args.scalibility_rio != 1):
        trainloader = Data_Scalability(trainset, args.scalibility_rio, BATCH_SIZE[args.dataset], mode=args.scalibility_mode, random_state=random_state, wild_dataset=wild_dataset) 

    # Training
    best_val_acc = 0.
    fname = f"{args.dataset}_{args.baseline}_1_{args.scalibility_rio}.txt"
    if(args.pretrained[0:4] != "clip" and args.baseline[0:4] == "CLIP"):
        raise Exception(f"{args.pretrained} not supported {args.baseline}")
    elif(args.baseline == "LP"):
        best_val_acc = LP(fname, model, args.pretrained, class_num, trainloader, testloader, args.epoch, args.lr, device, wild_dataset=wild_dataset)
    elif(args.baseline == "FF"):
        best_val_acc = Full_Finetune(fname, model, args.pretrained, class_num, trainloader, testloader, args.epoch, args.lr, device, wild_dataset=wild_dataset)
    elif(args.baseline == "Scartch"):   
        model = torchvision.models.resnet18(pretrained=False)
        best_val_acc = Full_Finetune(fname, model, args.pretrained, class_num, trainloader, testloader, args.epoch, args.lr, device, wild_dataset=wild_dataset)
    elif(args.baseline == "CLIP_TP"):  
        best_val_acc = CLIP_Pure(model, testloader, class_names, device, wild_dataset=wild_dataset)
    elif(args.baseline == "CLIP_LP" and args.pretrained == "clip_large"):  
        best_val_acc = CLIP_LP(fname, model, trainloader, testloader, class_num, args.epoch, args.lr, device, b_l="l", wild_dataset=wild_dataset)
    elif(args.baseline == "CLIP_LP" and args.pretrained == "clip"): 
        best_val_acc = CLIP_LP(fname, model, trainloader, testloader, class_num, args.epoch, args.lr, device, wild_dataset=wild_dataset)

    print("Best Validation Accuracy: ", best_val_acc)
    print("Execution Time (minutes): ", time.time() - start_time)
    



