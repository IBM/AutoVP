from auto_vp.wrapper import BaseWrapper
from auto_vp import programs
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD

from torchvision import transforms
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter
import os

def Load_Reprogramming_Model(dataset, device, file_path=None, mapping_method="frequency_based_mapping", set_train_resize=True, pretrained_model="resnet18", mapping=None, scale=1.0, num_map=1, weightinit=True):
    ##### Model Setting #####
    channel = 3
    img_resize = IMG_SIZE[dataset]
    class_num = CLASS_NUMBER[dataset]
    ##### End of Setting #####    

    if(file_path != None): # load from file_path
        # Load model # need more general
        state_dict = torch.load(file_path,  map_location=torch.device('cpu'))

        # Load resize module
        try:
            resize = programs.Trainable_Resize(output_size=(3, 224, 224)) 
            resize.load_state_dict(state_dict["resize_dict"])
        except:
            resize = None

        # redefind image size
        scale = state_dict["init_scale"]
        if(resize == None): 
            img_resize = int(img_resize*scale)
            if(img_resize > 224):
                img_resize = 224

        # Load pretrained model name and info
        pretrained_model = state_dict["pretrained_model"]

        if(pretrained_model == "clip_ViT_B_32"):
            normalization = None
            source_class_num = SOURCE_CLASS_NUM[pretrained_model]
            padding_size = None 
        elif(pretrained_model[0:4] == "clip"):
            normalization = None
            source_class_num = SOURCE_CLASS_NUM[pretrained_model] * class_num # 81 * class_num
            padding_size = None
        else:
            normalization = transforms.Normalize(mean=NETMEAN[pretrained_model], std=NETSTD[pretrained_model])
            source_class_num = SOURCE_CLASS_NUM[pretrained_model]
            padding_size = None

        # Load Input Perturbation
        input_pad = programs.InputPadding(img_size=(channel, img_resize, img_resize), output_size=(
            3, 224, 224), normalization=normalization, input_aware=False, padding_size=padding_size, model_name=pretrained_model, dataset_name=dataset, device=device)
        input_pad.load_state_dict(state_dict["perturb_dict"])

        # Load Output Mapping
        if(state_dict["output_mapping"] == None):
            num_source_to_map = 0
        else:
            num_source_to_map = len(state_dict["output_mapping"][0])

        out_map = programs.Output_Mapping(source_class_num=source_class_num, target_class_num=class_num,
                                        mapping_method=state_dict["mapping_method"], num_source_to_map=num_source_to_map, self_definded_map=state_dict["output_mapping"], weightinit=False, device=device)    
        out_map.load_state_dict(state_dict["outmap_dict"])
        out_map.freq_check = state_dict["freq_check"]

    else: # Build a new model
        # prepare mapping list
        if (mapping == None and mapping_method == "self_definded_mapping"):
            mapping_sequence = torch.randperm(SOURCE_CLASS_NUM[pretrained_model])[:class_num*num_map]
            mapping =  [[x.item() for x in mapping_sequence[i*num_map:(i+1)*num_map]] for i in range(class_num)]
            print(mapping) 

        # build model
        if(set_train_resize == True):
            resize = programs.Trainable_Resize(output_size=(3, 224, 224)) 
        else:
            resize = None

        if(resize != None):
            resize.scale = Parameter(torch.tensor(scale), requires_grad=True)
        else:
            # redefind image size
            img_resize = int(img_resize*scale)
            if(img_resize > 224):
                img_resize = 224

        if(pretrained_model == "clip_ViT_B_32"):
            normalization = None
            source_class_num = SOURCE_CLASS_NUM[pretrained_model]
            padding_size = None 
        elif(pretrained_model[0:4] == "clip"):
            normalization = None
            source_class_num = SOURCE_CLASS_NUM[pretrained_model] * class_num
            padding_size = None
        else:
            normalization = transforms.Normalize(mean=NETMEAN[pretrained_model], std=NETSTD[pretrained_model])
            source_class_num = SOURCE_CLASS_NUM[pretrained_model]
            padding_size = None

        input_pad = programs.InputPadding(img_size=(channel, img_resize, img_resize), output_size=(
            3, 224, 224), normalization=normalization, input_aware=False, padding_size=padding_size, model_name=pretrained_model, dataset_name=dataset, device=device)
        out_map = programs.Output_Mapping(source_class_num=source_class_num, target_class_num=class_num,
                                        mapping_method=mapping_method, num_source_to_map=num_map, self_definded_map=mapping, weightinit=weightinit, device=device) 

    reprogram_model = BaseWrapper(model_name=pretrained_model, dataset_name=dataset, input_perturbation=input_pad,
                                output_mapping=out_map, train_resize=resize, init_scale=scale, clip_img_size=img_resize, device=device)
    return reprogram_model