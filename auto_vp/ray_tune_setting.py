from auto_vp.utilities import setup_device
from auto_vp.const import MAP_NUMBER

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

def FreqLabelMap(model, trainloader, device, wild_dataset=False):
    if(model.output_mapping.mapping_method == "frequency_based_mapping"): 
        model.output_mapping.Frequency_mapping(model, trainloader, device, wild_dataset)
        #print(model.output_mapping.self_definded_map)
    return

def Training_local(model, trainloader, testloader, class_names, Epoch, lr, weight_decay, device, freqmap_interval=None, report=True, wild_dataset=False, convergence=False):    
    from auto_vp.wrapper import BaseWrapper
    from auto_vp import programs
    from auto_vp.imagenet1000_classname import IMGNET_CLASSNAME

    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader
    from sklearn.model_selection import KFold
    from torch.utils.data import ConcatDataset
    import matplotlib.pyplot as plt
    from torch.cuda.amp import autocast, GradScaler

    if(model.model_name[0:4] == "clip"):
        # Prepare text embedding
        template_number = 0 # use default template
        model.CLIP_Text_Embedding(class_names, template_number) 

    if(model.output_mapping.mapping_method == "semantic_mapping"):
        source_labels = list(IMGNET_CLASSNAME.values())
        model.output_mapping.Semantic_mapping(source_labels, class_names, show_map = False)

    # Update stretagy
    criterion = nn.CrossEntropyLoss()
    
    t_max = 200 * len(trainloader)
    if(model.model_name[0:4] == "clip"):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*t_max), int(0.72*t_max)], gamma=0.1)

    # Frequency mapping
    FreqLabelMap(model, trainloader, device, wild_dataset=wild_dataset)

    # Convergence loss
    MseLoss = nn.MSELoss(reduction='sum')
    loss2 = torch.zeros([]).to(device)
    if(model.model_name[0:4] == "clip"):
        # Convergence loss: init condition for fully connected mapping
        init_weight = torch.zeros([model.output_mapping.target_class_num, model.output_mapping.target_class_num*81]).to(device) # torch.Size([#_class, 81*#_class])
        for i in range(model.output_mapping.target_class_num):
            init_weight[i, i] = 1.0
        zero_bias = torch.zeros([model.output_mapping.target_class_num]).to(device) # torch.Size([#_class])

        # Convergence loss: init condition for frequency mapping
        if(model.output_mapping.mapping_method == "frequency_based_mapping" and freqmap_interval != None):
            freq_conv_loss_update = 1
            init_freq_mapping = torch.zeros([model.output_mapping.target_class_num, model.output_mapping.target_class_num*81]).to(device)
            for i in range(model.output_mapping.target_class_num):
                for j in model.output_mapping.self_definded_map[i]:
                    init_freq_mapping[i, j] = 1.0

            current_freq_mapping = torch.zeros([model.output_mapping.target_class_num, model.output_mapping.target_class_num*81]).to(device)
            for i in range(model.output_mapping.target_class_num):
                for j in model.output_mapping.self_definded_map[i]:
                    current_freq_mapping[i, j] = 1.0

    best_result = [-1, 0., 0., 1.] # epoch, traing acc, validation acc, resize scale
    total_train_acc = 0
    total_valid_acc = 0

    if(model.model_name[0:4] == "clip"):
        scaler = GradScaler()
    for epoch in range(Epoch):
        # Frequency mapping
        if(model.output_mapping.mapping_method == "frequency_based_mapping" and freqmap_interval != None and epoch!= 0 and epoch%freqmap_interval == 0):
            FreqLabelMap(model, trainloader, device, wild_dataset=wild_dataset)
            freq_conv_loss_update = 1
            if(model.model_name[0:4] == "clip"):
                current_freq_mapping = torch.zeros([model.output_mapping.target_class_num, model.output_mapping.target_class_num*81]).to(device)
                for i in range(model.output_mapping.target_class_num):
                    for j in model.output_mapping.self_definded_map[i]:
                        current_freq_mapping[i, j] = 1.0

        # Training
        model.train()
        model.model.eval()
        train_loss = []
        train_loss2 = []
        train_accs = []
        pbar = tqdm(trainloader, total=len(trainloader),
                    desc=f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=160)

        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            if imgs.get_device() == -1:
                imgs = imgs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
                if(model.model_name[0:4] == "clip" and convergence == True):
                    if(model.output_mapping.mapping_method == "fully_connected_layer_mapping"):
                        loss2 = (MseLoss(model.output_mapping.layers.weight, init_weight) + MseLoss(model.output_mapping.layers.bias, zero_bias)) * loss2_weight 
                        loss += loss2
                    elif(model.output_mapping.mapping_method == "frequency_based_mapping" and freqmap_interval != None and freq_conv_loss_update == 1):
                        loss2 = MseLoss(current_freq_mapping, init_freq_mapping) * loss2_weight 
                        loss += loss2
                        freq_conv_loss_update = 0

            if(model.model_name[0:4] == "clip"):
                scaler.scale(loss).backward()
                # clip scale's gradient
                if(model.no_trainable_resize == 0): 
                    nn.utils.clip_grad_value_(model.train_resize.scale, 0.001) # 0.0001

                if(model.output_mapping.mapping_method == "fully_connected_layer_mapping"):
                    nn.utils.clip_grad_value_(model.output_mapping.layers.bias, 0.001)
                    nn.utils.clip_grad_value_(model.output_mapping.layers.weight, 0.001)
                
                scaler.step(optimizer)
                scaler.update()

                model.model.logit_scale.data = torch.clamp(model.model.logit_scale.data, 0, 4.6052)
            else:
                loss.backward()
                optimizer.step()

            if(model.no_trainable_resize == 0):
                with torch.no_grad():
                    model.train_resize.scale = model.train_resize.scale.clamp_(0.1, 5.0)

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_loss2.append(loss2.item())
            train_accs.append(acc)

            total_train_loss = sum(train_loss) / len(train_loss)
            total_train_loss2 = sum(train_loss2) / len(train_loss2)
            total_train_acc = sum(train_accs) / len(train_accs)
            if(model.no_trainable_resize == 0):
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Loss2: {total_train_loss2:.4f}, Scale: {model.train_resize.scale.item():.4f}")
            else:
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Loss2: {total_train_loss2:.4f}")

            scheduler.step()
          
        if total_train_acc > best_result[2]:
            ss = 1.
            if(model.no_trainable_resize == 0):
                ss = model.train_resize.scale.item()
            best_result = [epoch, total_train_acc, 0, ss]
        
        if(report == True):
            ### Tune report ###
            if(model.no_trainable_resize == 0):
                scale = model.train_resize.scale.item()
            else:
                scale = model.init_scale
            tune.report(accuracy=total_train_acc.cpu().item(), last_scale=scale)
  
    return best_result

def train_model(config, dataset, data_path, download=True, scalibility_rio=1, scalibility_mode="equal", wild_dataset=False, convergence=False, LRWD=False): 
    from auto_vp.wrapper import BaseWrapper
    from auto_vp.dataprepare import DataPrepare, Data_Scalability
    from auto_vp.utilities import setup_device
    from auto_vp import programs
    import auto_vp.datasets as datasets
    from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD, RAY_BATCH_SIZE

    from torch.nn.parameter import Parameter

    # device setting
    device, list_ids = setup_device(1)
    print("device: ", device)

    ### config ###
    mapping_method = config["mapping_method"]
    num_map = config["num_map"]
    freqmap_interval = config["freqmap_interval"]
    scale = config["scale"]
    set_train_resize = config["set_train_resize"]
    pretrained_model = config["pretrained_model"]

    if(LRWD == False):
        if(pretrained_model[0:4] == "clip"):
            lr = 40
        else:
            lr = 0.001
        weight_decay = 0.0
    else:
        lr = config["lr"]
        weight_decay = config["weight_decay"]
    ###############

    # Dataset Setting
    channel = 3
    img_resize = IMG_SIZE[dataset]
    class_num = CLASS_NUMBER[dataset]
    random_state = 1
    
    # prepare mapping list
    mapping = None 
    if (mapping_method == "self_definded_mapping"):
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
        3, 224, 224), normalization=normalization, input_aware=False, padding_size=padding_size, model_name=pretrained_model, device=device)
    out_map = programs.Output_Mapping(source_class_num=source_class_num, target_class_num=class_num,
                                      mapping_method=mapping_method, num_source_to_map=num_map, self_definded_map=mapping, device=device) 
    reprogram_model = BaseWrapper(model_name=pretrained_model, input_perturbation=input_pad,
                                  output_mapping=out_map, train_resize=resize, init_scale=scale, clip_img_size=img_resize, device=device)

    if(pretrained_model[0:4] == "clip"):
        clip_transform = reprogram_model.clip_preprocess
    else:
        clip_transform = None

    # dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=dataset, dataset_dir=data_path, target_size=(
        img_resize, img_resize), mean=NETMEAN[pretrained_model], std=NETSTD[pretrained_model], download=download, batch_size=RAY_BATCH_SIZE[dataset], random_state=random_state, clip_transform=clip_transform)

    if(scalibility_rio != 1):
        trainloader = Data_Scalability(trainset, scalibility_rio, BATCH_SIZE[dataset], mode=scalibility_mode, random_state=random_state, wild_dataset=wild_dataset) 

    # Training
    best_result = Training_local(model=reprogram_model, trainloader=trainloader, testloader=testloader, class_names=class_names, Epoch=5, lr=lr, weight_decay=weight_decay, device=device, freqmap_interval=freqmap_interval, report=True, wild_dataset=wild_dataset, convergence=convergence) 

    print("Finished Training")

def Config_Setting(mapping_type, num_map_range, scale_range):
    if(mapping_type == "frequency_based_mapping"):
        config = {
            "mapping_method": tune.choice(["frequency_based_mapping"]),
            "num_map":  tune.grid_search(num_map_range), 
            "freqmap_interval": tune.grid_search([None, 1]),  
            "scale": tune.grid_search(scale_range), 
            "set_train_resize": tune.grid_search([True, False]),
            "pretrained_model" : tune.grid_search(["resnet18", "ig_resnext101_32x8d", "swin_t", "clip"]) 
        }
    elif(mapping_type == "semantic_mapping"):
        config = {
            "mapping_method": tune.choice(["semantic_mapping"]), 
            "num_map": tune.grid_search(num_map_range), 
            "freqmap_interval": tune.choice([None]),
            "scale": tune.grid_search(scale_range),
            "set_train_resize": tune.grid_search([True, False]), 
            "pretrained_model" :  tune.grid_search(["resnet18", "ig_resnext101_32x8d", "swin_t"]) # No "clip" !
        }
    elif(mapping_type == "fully_connected_layer_mapping"):
        config = {
            "mapping_method": tune.choice(["fully_connected_layer_mapping"]), 
            "num_map": tune.choice([None]),
            "freqmap_interval": tune.choice([None]),
            "scale": tune.grid_search(scale_range), 
            "set_train_resize": tune.grid_search([True, False]),
            "pretrained_model" : tune.grid_search(["resnet18", "ig_resnext101_32x8d", "swin_t", "clip"])
        }
    else:
        raise NotImplementedError(f"Ray Tune Config: mapping_type: {mapping_type} not supported")
    
    return config


def Parameter_Tune(dataset, data_path, download=True, scalibility_rio=1, scalibility_mode="equal", wild_dataset=False, convergence=False): # , mapping_method, OutMap_num, scale
    # ref: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    # ref: https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg
    from auto_vp.const import RAY_MAX_EPOCH, RAY_MIN_EPOCH
    from auto_vp.dataprepare import DataPrepare
    import os

    num_map_range = MAP_NUMBER[dataset]
    num_map_range = [int(x) for x in num_map_range]
    scale_range = [0.5, 1.0, 1.5] 
    scale_range = [float(x) for x in scale_range] # cannot be numpy.float64
    print(scale_range)
    print(num_map_range)

    Best_trail_acc = []
    Best_trail_setting = []

    file_name = f"{dataset}_ray_tune_1_{scalibility_rio}.csv"
    f = open(file_name,  "w+")
    f.close()

    for mapping_type in ["fully_connected_layer_mapping", "semantic_mapping", "frequency_based_mapping"]: 
        config = Config_Setting(mapping_type, num_map_range, scale_range)

        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=RAY_MAX_EPOCH[dataset],
            grace_period=RAY_MIN_EPOCH[dataset],
            reduction_factor=2)

        reporter = CLIReporter(
            metric_columns=["accuracy", "training_iteration", "last_scale"])

        if(download == True): # just for download
            trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=dataset, dataset_dir=data_path, target_size=(
                128, 128), download=download, batch_size=128, random_state=7)

        if(data_path[0] != "/"): # not from root 
            data_path = os.path.join(os.getcwd(), data_path)
        #os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
        result = tune.run(
            partial(train_model, dataset=dataset, data_path=data_path,  download=False, scalibility_rio=scalibility_rio, scalibility_mode=scalibility_mode, wild_dataset=wild_dataset, convergence=convergence),
            config=config,
            resources_per_trial={"cpu": 2, "gpu": 1},
            scheduler=scheduler,
            progress_reporter=reporter)


        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Mapping Type: ", mapping_type)
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final training acc: {}".format(best_trial.last_result["accuracy"]))

        # Save the tuning result table  
        df = result.results_df[['accuracy', 'training_iteration', 'config/mapping_method', 'config/num_map', 'config/freqmap_interval', 'config/scale', 'config/set_train_resize', 'last_scale', 'config/pretrained_model']]
        df.to_csv(file_name, index=False, header=True, mode='a', encoding="utf_8_sig")

        # Save the best trial config info
        Best_trail_acc.append(best_trial.last_result["accuracy"])
        Best_trail_setting.append(best_trial.config)
    
    # Get the best trial config
    best_trial_cfg = Best_trail_setting[np.argmax(Best_trail_acc)]

    return best_trial_cfg["mapping_method"], best_trial_cfg["num_map"],  best_trial_cfg["freqmap_interval"], best_trial_cfg["scale"], best_trial_cfg["set_train_resize"], best_trial_cfg["pretrained_model"]

def Config_Setting_LR_WD(lr_list, weight_decay_list, mapping_method, num_map, freqmap_interval, scale, set_train_resize, pretrained_model):
    config = {
            "mapping_method": tune.choice([mapping_method]), 
            "num_map": tune.choice([num_map]),
            "freqmap_interval": tune.choice([freqmap_interval]),
            "scale": tune.choice([float(scale)]), 
            "set_train_resize": tune.choice([set_train_resize]),
            "pretrained_model" : tune.choice([pretrained_model]),
            "lr" : tune.grid_search(lr_list), 
            "weight_decay" : tune.grid_search(weight_decay_list), 
        }
    return config


def Parameter_Tune_LRWD(dataset, data_path, mapping_method, num_map, freqmap_interval, scale, set_train_resize, pretrained_model, download=False, scalibility_rio=1, scalibility_mode="equal", wild_dataset=False, convergence=False):
    from auto_vp.const import RAY_MAX_EPOCH, RAY_MIN_EPOCH
    import os
    if(pretrained_model[0:4] == "clip"):
        lr_list = [35, 40, 45]
    else:
        lr_list = [1e-2, 1e-3, 1e-4]
    lr_list = [float(x) for x in lr_list]
    weight_decay_list = [0.0, 1e-5, 1e-10]

    Best_trail_acc = []
    Best_trail_setting = []

    file_name = f"{dataset}_ray_tune_1_{scalibility_rio}_LR_WD.csv"
    f = open(file_name,  "w+")
    f.close()

    config = Config_Setting_LR_WD(lr_list, weight_decay_list, mapping_method, num_map, freqmap_interval, scale, set_train_resize, pretrained_model)

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=RAY_MAX_EPOCH[dataset],
        grace_period=RAY_MIN_EPOCH[dataset],
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["accuracy", "training_iteration", "last_scale"])

    if(download == True): # just for download
        trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=dataset, dataset_dir=data_path, target_size=(
            128, 128), download=download, batch_size=128, random_state=7)

    if(data_path[0] != "/"): # not from root 
        data_path = os.path.join(os.getcwd(), data_path)
    #os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
    result = tune.run(
        partial(train_model, dataset=dataset, data_path=data_path,  download=False, scalibility_rio=scalibility_rio, scalibility_mode=scalibility_mode, wild_dataset=wild_dataset, convergence=convergence, LRWD=True),
        config=config,
        resources_per_trial={"cpu": 2, "gpu": 1},
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final training acc: {}".format(best_trial.last_result["accuracy"]))

    # Save the tuning result table  
    df = result.results_df[['accuracy', 'training_iteration', 'config/lr', 'config/weight_decay', 'config/mapping_method', 'config/num_map', 'config/freqmap_interval', 'config/scale', 'config/set_train_resize', 'last_scale', 'config/pretrained_model']]
    df.to_csv(file_name, index=False, header=True, mode='a', encoding="utf_8_sig")

    # Save the best trial config info
    best_trial_cfg = best_trial.config 

    return best_trial_cfg["lr"], best_trial_cfg["weight_decay"]
