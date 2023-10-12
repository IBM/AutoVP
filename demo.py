import auto_vp.datasets as datasets
from auto_vp.utilities import setup_device, Trainable_Parameter_Size
from auto_vp.dataprepare import DataPrepare, Data_Scalability
from auto_vp import programs
from auto_vp.training_process import *
from auto_vp.ray_tune_setting import Parameter_Tune, Parameter_Tune_LRWD
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD
from auto_vp.load_model import Load_Reprogramming_Model

import argparse
from torchvision import transforms
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter
import os

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
    
    p.add_argument('--param_tune', type=int, choices=[0, 1], default=0) 
    p.add_argument('--LR_WD_tune', type=int, choices=[0, 1], default=0) 
    p.add_argument(
        '--pretrained', choices=["vgg16_bn", "resnet18", "resnet50", "resnext101_32x8d", "ig_resnext101_32x8d", "vit_b_16", "swin_t", "clip", "clip_large", "clip_ViT_B_32"], default="swin_t")
    

    p.add_argument('--mapping_method', choices=["fully_connected_layer_mapping", "frequency_based_mapping", "self_definded_mapping", "semantic_mapping"], default="fully_connected_layer_mapping")
    p.add_argument('--img_scale', type=float, default=1.5) 
    p.add_argument('--out_map_num', type=int, default=5) 
    p.add_argument('--train_resize', type=int, default=0) # 1, 0
    p.add_argument('--freqmap_interval', type=int, default=-1) # -1 or 1,2,3..
    p.add_argument('--weightinit', type=int, default=1) # 1, 0 

    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001) 
    p.add_argument('--seed', type=int, default=7)

    p.add_argument('--scalibility_rio', type=int, choices=[1, 2, 4, 10, 100], default=1) 
    p.add_argument('--scalibility_mode', choices=["equal", "random"], default="equal") 

    start_time = time.time()

    args = p.parse_args()
    
    # set random seed
    set_seed(args.seed)

    # device setting
    device, list_ids = setup_device(1)
    print("device: ", device)

    # Create datapath directory
    isExist = os.path.exists(args.datapath)
    if not isExist:
        os.makedirs(args.datapath)
        print(f"Create dir {args.datapath}")

    ##### Model Setting #####
    # Modelure functionality
    # choice: [True, False]. Turn on/off the ray tune
    if(args.param_tune == 0):
        param_tune = False
    else:
        param_tune = True  
    
    # choice: [True, False]. Turn on/off the ray tune with additional learning rate (LR)/ weight decay (WD) options
    if(args.LR_WD_tune == 0):
        LR_WD_tune = False
    else:
        LR_WD_tune = True  

    # choice: [True, False]. Turn on/off the trainable resize module
    if(args.train_resize > 0):
        set_train_resize = True
    else:
        set_train_resize = False
    
    # choice: [None, $INT.]. Turn on/off the iterative freq mapping
    if(args.freqmap_interval < 1):
        freqmap_interval = None
    else:
        freqmap_interval = args.freqmap_interval     
    
    if(args.weightinit > 0):
        weightinit = True
    else:
        weightinit = False
    
    # choice: [True, False]. Download the dataset or not.
    if(args.download  > 0):
        download  = True
    else:
        download  = False


    # Reprogramming Model Setting
    mapping_method = args.mapping_method # choice: ["fully_connected_layer_mapping", "frequency_based_mapping", "self_definded_mapping", "semantic_mapping"]
    scale = args.img_scale
    num_map = args.out_map_num

    # Dataset Setting and Model Loading  
    file_path = None        # choice: [None, f"{args.dataset}_best.pth"] # Create a new module or load from ckpt 
    random_state = args.seed        # for some dataset in DataPrepare() and Data_Scalability()
    ##### End of Setting #####

    img_resize = IMG_SIZE[args.dataset]
    pretrained_model = args.pretrained

    wild_ds_list = ["Camelyon17", "Iwildcam", "FMoW"]
    if args.dataset in wild_ds_list:
        wild_dataset = True
    else:
        wild_dataset = False

    # Tune parameter
    file_name = f"{args.dataset}_log_1_{args.scalibility_rio}.txt"
    f = open(file_name,  "w+")
    if(param_tune == True):
        print("Warning: If you turn on param_tune, then the arguments will be ignored!")
        mapping_method, num_map, freqmap_interval, scale, set_train_resize, pretrained_model = Parameter_Tune(dataset=args.dataset, data_path=args.datapath, download=download, scalibility_rio=args.scalibility_rio, scalibility_mode=args.scalibility_mode, wild_dataset=wild_dataset)
        print(f"Ray Tune result: mapping_method={mapping_method}, num_map={num_map}, freqmap_interval={freqmap_interval}, scale={scale}, set_train_resize={set_train_resize}, pretrained_model={pretrained_model}")
        f.write(f"Ray Tune result: mapping_method={mapping_method}, num_map={num_map}, freqmap_interval={freqmap_interval}, scale={scale}, set_train_resize={set_train_resize}, pretrained_model={pretrained_model}\n")
    else:
        f.write(f"NO Tuning: mapping_method={mapping_method}, num_map={num_map}, freqmap_interval={freqmap_interval}, scale={scale}, set_train_resize={set_train_resize}, pretrained_model={pretrained_model}\n")
    
    # LR/WD Tuning
    if(LR_WD_tune == True):
        lr, weight_decay = Parameter_Tune_LRWD(dataset=args.dataset, data_path=args.datapath, mapping_method=mapping_method, num_map=num_map, freqmap_interval=freqmap_interval, scale=scale, set_train_resize=set_train_resize, pretrained_model=pretrained_model, download=download, scalibility_rio=args.scalibility_rio, scalibility_mode=args.scalibility_mode, wild_dataset=wild_dataset)
        f.write(f"LR/WD Ray Tune result: lr={lr}, weight_decay={weight_decay}\n")
    else:
        lr = args.lr
        weight_decay = 0.0
        f.write(f"LR/WD NO Tuning: lr={lr}, weight_decay={weight_decay}\n")

    f.write(f"Tuning Time (second) : %s" % (time.time() - start_time))
    f.close()

    
    # Load or build a reprogramming model
    reprogram_model = Load_Reprogramming_Model(args.dataset, device, file_path=file_path, mapping_method=mapping_method, set_train_resize=set_train_resize, pretrained_model=pretrained_model, mapping=None, scale=scale, num_map=num_map, weightinit=weightinit)
    Trainable_Parameter_Size(reprogram_model, file_name)
    
    if(reprogram_model.model_name[0:4] == "clip"):
        clip_transform = reprogram_model.clip_preprocess
    else:
        clip_transform = None

    # redefind image size
    if(set_train_resize == False):
        img_resize = int(img_resize*scale)
        if(img_resize > 224):
            img_resize = 224

    # Dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath, target_size=(
        img_resize, img_resize), mean=NETMEAN[reprogram_model.model_name], std=NETSTD[reprogram_model.model_name], download=download, batch_size=BATCH_SIZE[args.dataset], random_state=random_state, clip_transform=clip_transform)
    
    if(args.scalibility_rio != 1):
        trainloader = Data_Scalability(trainset, args.scalibility_rio, BATCH_SIZE[args.dataset], mode=args.scalibility_mode, random_state=random_state, wild_dataset=wild_dataset) 

    # Training
    fname = f"{args.dataset}_log_1_{args.scalibility_rio}.txt"
    if(pretrained_model[0:4] == "clip"):
        CLIP_Training(args.dataset, fname, reprogram_model, trainloader, testloader, class_names, args.epoch, lr, weight_decay, device, freqmap_interval=freqmap_interval, wild_dataset=wild_dataset) # , convergence=True 
    else:
        Training(args.dataset, fname, reprogram_model, trainloader, testloader, class_names, args.epoch, lr,  weight_decay, device, freqmap_interval=freqmap_interval, wild_dataset=wild_dataset)

    f = open(file_name,  "a")
    f.write(f"Total Exection Time (second) : %s" % (time.time() - start_time))
    f.close()

