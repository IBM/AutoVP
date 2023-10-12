from auto_vp.wrapper import BaseWrapper
from auto_vp.utilities import setup_device
from auto_vp.dataprepare import DataPrepare, Data_Scalability
from auto_vp import programs
from auto_vp.training_process import Training
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD
from auto_vp.load_model import Load_Reprogramming_Model

import argparse
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import manifold
from torch.nn import functional as F
import cv2
import os

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

def t_SNE(model, pretrained_model, trainloader, testloader):
    # Step 1: Get Embedding vector from pretrained model avgpool
    features = {}
    feat = []
    def get_features(name):
        def hook(model, input, output):
            features['feats'] = output.detach()
        return hook

    feature_dim = None
    if(pretrained_model == "resnet18"): 
        model.model.avgpool.register_forward_hook(get_features('feats'))
        feature_dim = 512
    elif(pretrained_model == "resnet50"): 
        model.model.avgpool.register_forward_hook(get_features('feats'))
        feature_dim = 2048
    elif(pretrained_model == "resnext101_32x8d"): 
        model.model.avgpool.register_forward_hook(get_features('feats'))
        feature_dim = 2048
    elif(pretrained_model == "vgg16_bn"): 
        model.model.classifier[3].register_forward_hook(get_features('feats'))
        feature_dim = 4096
    elif(pretrained_model == "vit_b_16"): 
        model.model.encoder.ln.register_forward_hook(get_features('feats'))
        feature_dim = 768
    else:
        raise NotImplementedError(f"{pretrained_model} not supported")

    model.eval()
    train_labels = []
    for batch in tqdm(trainloader):
        imgs, labels = batch
        train_labels.append(labels)
        for img in imgs:
            img = torch.unsqueeze(img, axis=0)
            logits = model(img.to(device))
            feat.append(features['feats'].cpu()) # len(features['feats'][0]) -> features dim
    train_len = len(feat)

    test_labels = []
    for batch in tqdm(testloader):
        imgs, labels = batch
        test_labels.append(labels)
        for img in imgs:
            img = torch.unsqueeze(img, axis=0)
            logits = model(img.to(device))
            feat.append(features['feats'].cpu())
        #if(len(test_labels) > 10):
        #    break
    test_len = len(feat) - train_len
    print(f"Train len: {train_len}, Test len: {test_len}")

    # Step 2: manifold.TSNE
    feat_2 = []
    for a in feat: 
        temp = []
        # a in vit_b_16 : [1, 197, 768] -> [1, # of patches, feature dim]
        # a in resnext101_32x8d : [1, 2048, 1, 1] 
        for j in range(feature_dim): # dim of avgpool
            if (pretrained_model == "vit_b_16"): 
                # ref: https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/3
                # Classifier "token" as used by standard language architectures
                temp.append(a[0][0][j].item())
            else:
                temp.append(a[0][j][0][0].item())
        feat_2.append(temp)

    feat_2 = np.array(feat_2)
    print(feat_2.shape)
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(feat_2)

    # Normalization the processed features 
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # Data Visualization
    # Training Data
    tt = []
    for x in train_labels:
        for y in x:
            tt.append(y)

    plt.figure(dpi=350)
    plt.scatter(X_norm[0:train_len, 0], X_norm[0:train_len, 1], c=tt, s=1, cmap='gist_ncar') # 'tab10'
    plt.colorbar()
    plt.grid()
    plt.savefig("Train_tsne.png")

    # Testing Data
    tt = []
    for x in test_labels:
        for y in x:
            tt.append(y)

    plt.figure(dpi=350)
    plt.scatter(X_norm[train_len::, 0], X_norm[train_len::, 1], c=tt, s=1, cmap='gist_ncar') # 'tab10'
    plt.colorbar()
    plt.grid()
    plt.savefig("Test_tsne.png")

# ref: https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
# ref: https://debuggercafe.com/basic-introduction-to-class-activation-maps-in-deep-learning-using-pytorch/
def returnCAM(feature_conv, weight_softmax, idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    print(bz, nc, h, w)

    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    # print(cam_img.shape) # (7, 7)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img

def Heat_Map(model, pretrained_model, imgs, labels, class_names, device):
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    
    if(pretrained_model == "resnet18"): 
        model.model._modules.get('layer4').register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(model.model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    model.model.requires_grad_(True) 
    model.eval()
    fig, ax = plt.subplots(4, 2, dpi=350, figsize=(10,20))
    i = 0
    for img, lab in zip(imgs, labels):
        img = torch.unsqueeze(img, axis=0)


        img_h = -1
        img_w = -1
        if(model.no_trainable_resize == 0):
            logit, img_h, img_w = model.train_resize(img.to(device))
        else:
            logit = model.train_resize(img)
        logit_pur = model.input_perturbation(logit, img_h, img_w)

        if(model.model_name[0:4] == "clip"):
            logit = model.CLIP_network(logit_pur)
        else:
            logit = model.model(logit_pur)
        
        logit = model.output_mapping(logit)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[i], weight_softmax, [idx[0]])

        # render the CAM and output
        aa = np.transpose(logit_pur[0].cpu().detach().numpy(), (1, 2, 0))
        ax[i, 1].imshow(aa)
        ax[i, 1].contourf(CAMs, alpha=0.6, cmap=plt.cm.jet)

        ax[i, 0].imshow(aa)
        ax[i, 0].set_title("class: "+str(class_names[lab])+", predicted class: "+str(class_names[idx[0]]))
        i += 1
        if(i == 4):
            break
        
    plt.savefig('CAM.jpg')

    return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset', choices=["CIFAR10", "CIFAR10-C", "CIFAR100", "Melanoma", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "UCF101", "FMoW"], required=True)
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--download', type=int, choices=[0, 1], default=0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--scalibility_rio', type=int, choices=[1, 2, 4, 10, 100], default=1)
    p.add_argument('--scalibility_mode', choices=["equal", "random"], default="equal")  

    args = p.parse_args()

    # set random seed
    set_seed(args.seed)

    # device setting
    device, list_ids = setup_device(1)
    print("device: ", list_ids)

    ##### Model Setting #####
    channel = 3
    img_resize = IMG_SIZE[args.dataset]
    class_num = CLASS_NUMBER[args.dataset]
    random_state = args.seed
    ##### End of Setting #####  

    # choice: [True, False]. Download the dataset or not.
    if(args.download  > 0):
        download  = True
    else:
        download  = False  

    # Load model 
    reprogram_model = Load_Reprogramming_Model(args.dataset, device, file_path=f"{args.dataset}_last.pth") 
    if(reprogram_model.no_trainable_resize == 1):
        set_train_resize = False
    else:
        set_train_resize = True

    print("Model Info:")
    print("set_train_resize: ", set_train_resize)
    print(reprogram_model.model_name)
    print(reprogram_model.output_mapping.mapping_method)


    if(reprogram_model.model_name[0:4] == "clip"):
        clip_transform = reprogram_model.clip_preprocess
    else:
        clip_transform = None

    scale = reprogram_model.init_scale
    print(scale)
    if(set_train_resize == False):
        # redefind image size
        img_resize = int(img_resize*scale)
        if(img_resize > 224):
            img_resize = 224
    
    # print(reprogram_model.output_mapping.self_definded_map)
    if(set_train_resize == True):
        print(reprogram_model.train_resize.scale)

    wild_ds_list = ["Camelyon17", "Iwildcam", "FMoW"]
    if args.dataset in wild_ds_list:
        wild_dataset = True
    else:
        wild_dataset = False
            
    # dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath, target_size=(
        img_resize, img_resize), mean=NETMEAN[reprogram_model.model_name], std=NETSTD[reprogram_model.model_name], download=download, batch_size=BATCH_SIZE[args.dataset], random_state=random_state, clip_transform=clip_transform)

    if(args.scalibility_rio != 1):
        trainloader = Data_Scalability(trainset, args.scalibility_rio, BATCH_SIZE[args.dataset], mode=args.scalibility_mode, random_state=random_state, wild_dataset=wild_dataset) 
    # Plot t_SNE result    
    # t_SNE(reprogram_model, reprogram_model.model_name, trainloader, testloader)

    # Prepare text embedding
    if(reprogram_model.model_name[0:4] == "clip"):
        template_number = 0 # use default template
        reprogram_model.CLIP_Text_Embedding(class_names, template_number) 

    reprogram_model.eval()
    valid_loss = []
    valid_accs = []
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(testloader, total=len(testloader), desc=f"Testing", ncols=100)
    for pb in pbar:
        if(wild_dataset == True):
            imgs, labels, _ = pb
        else:
            imgs, labels = pb

        with torch.no_grad():
            logits = reprogram_model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

        total_valid_loss = sum(valid_loss) / len(valid_loss)
        total_valid_acc = sum(valid_accs) / len(valid_accs)
        pbar.set_postfix_str(f"ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}")
        
    # Plot the perturbation result
    for pb in pbar:
        if(wild_dataset == True):
            x, y, _ = pb
        else:
            x, y = pb

        with torch.no_grad():
            # clip need to resize by ourself 
            xx = reprogram_model.clip_rz_transform(x.to(device))
            a = -1
            b = -1
            if(set_train_resize == True):
                ims, a, b = reprogram_model.train_resize(xx)
            else:
                ims = reprogram_model.train_resize(xx)
            ims = reprogram_model.input_perturbation(ims, a, b)
        break
           
    fig, ax = plt.subplots(1, 4, dpi=350, figsize=(20,5))
    for i in range(4):
        im = ax[i].imshow(np.transpose(ims[i].cpu().detach().numpy(), (1, 2, 0)))
        ax[i].set_title(y[i].item())
        plt.colorbar(im) 
    plt.savefig(args.dataset+"_prompted_img.png")



    