from auto_vp.wrapper import BaseWrapper
from auto_vp import programs
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD, DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES
from auto_vp.imagenet1000_classname import IMGNET_CLASSNAME
from auto_vp.utilities import Trainable_Parameter_Size

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.cuda.amp import autocast, GradScaler
import clip

def FreqLabelMap(model, trainloader, device,  wild_dataset=False):
    if(model.output_mapping.mapping_method == "frequency_based_mapping"): # model.no_output_mapping != 1 and 
        model.output_mapping.Frequency_mapping(model, trainloader, device, wild_dataset)
        #print(model.output_mapping.self_definded_map)
    return

def Training(dataset, fname, model, trainloader, testloader, class_names, Epoch, lr, weight_decay, device, freqmap_interval=None, wild_dataset=False):
    if(model.output_mapping.mapping_method == "semantic_mapping"):
        source_labels = list(IMGNET_CLASSNAME.values())
        model.output_mapping.Semantic_mapping(source_labels, class_names)

    # Update stretagy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
       optimizer, milestones=[int(0.5*Epoch), int(0.72*Epoch)], gamma=0.1)

    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Frequency mapping
    FreqLabelMap(model, trainloader, device)

    f = open(fname, "a")
    best_result = [-1, 0., 0., 1.] # epoch, traing acc, validation acc, resize scale
    total_train_acc = 0
    total_valid_acc = 0
    scale_grad = []
    for epoch in range(Epoch):
        # Frequency mapping
        if(freqmap_interval != None and epoch!= 0 and epoch%freqmap_interval == 0):
            FreqLabelMap(model, trainloader, device)

        # Training
        model.train()
        model.model.eval()
        train_loss = []
        train_accs = []
        pbar = tqdm(trainloader, total=len(trainloader),
                    desc=f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=120)
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
            loss.backward()

            optimizer.step()

            if(model.no_trainable_resize == 0):
                with torch.no_grad():
                    model.train_resize.scale = model.train_resize.scale.clamp_(0.1, 5.0)

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

            total_train_loss = sum(train_loss) / len(train_loss)
            total_train_acc = sum(train_accs) / len(train_accs)
            if(model.no_trainable_resize == 0):
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Scale: {model.train_resize.scale.item():.4f}")
            else:
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}")
        
        # update log
        if(model.no_trainable_resize == 0):
            f.write(f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}, ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Scale: {model.train_resize.scale.item():.4f}\n")
        else:
            f.write(f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}, ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}\n")
        scheduler.step()

    
        if(epoch%10 ==0 or epoch == Epoch-1): 
            # Validation
            model.eval()
            valid_loss = []
            valid_accs = []
            pbar = tqdm(testloader, total=len(
                testloader), desc=f"Epoch {epoch+1} Testing", ncols=120)
            for pb in pbar:
                if(wild_dataset == True):
                    imgs, labels, _ = pb
                else:
                    imgs, labels = pb
                    
                if imgs.get_device() == -1:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                with torch.no_grad():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()

                valid_loss.append(loss.item())
                valid_accs.append(acc)

                total_valid_loss = sum(valid_loss) / len(valid_loss)
                total_valid_acc = sum(valid_accs) / len(valid_accs)
                pbar.set_postfix_str(f"ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}")

            # update log
            f.write(f"Epoch {epoch+1} Testing, ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}\n")

            if (total_valid_acc > best_result[2] or epoch == Epoch - 1):
                ss = None
                if(model.no_trainable_resize == 0):
                    ss = model.train_resize.scale.item()
                # save model
                print("Save! Acc: ", total_valid_acc, ", Scale: ", ss)
                f.write(f"Save! Acc: {total_valid_acc}, Scale: {ss}\n")
                state_dict = {
                    "pretrained_model": model.model_name,
                    "resize_dict": model.train_resize.state_dict(),
                    "perturb_dict": model.input_perturbation.state_dict(),
                    "outmap_dict": model.output_mapping.state_dict(),
                    "output_mapping": model.output_mapping.self_definded_map,
                    "mapping_method": model.output_mapping.mapping_method,
                    "freq_check": model.output_mapping.freq_check,
                    "optimizer_dict": optimizer.state_dict(),
                    "init_scale" : model.init_scale
                }
                if(total_valid_acc > best_result[2]):
                    best_result = [epoch, total_train_acc, total_valid_acc, ss]
                    torch.save(state_dict, str(dataset) + "_best.pth")
                if(epoch == Epoch - 1):
                    torch.save(state_dict, str(dataset) + "_last.pth")

    f.close()

    return best_result

# https://github.com/openai/CLIP
def CLIP_Training(dataset, fname, model, trainloader, testloader, class_names, Epoch, lr, weight_decay, device, freqmap_interval=None, wild_dataset=False, convergence=False):
    if(model.output_mapping.mapping_method == "semantic_mapping"):
        print("CLIP not support semantic mapping!")
        return

    # Prepare text embedding
    template_number = 0 # use default template
    model.CLIP_Text_Embedding(class_names, template_number) 

    # loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    t_max = Epoch * len(trainloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Frequency mapping
    FreqLabelMap(model, trainloader, device, wild_dataset=wild_dataset)

    # Convergence loss
    MseLoss = nn.MSELoss(reduction='sum')
    loss2 = torch.zeros([]).to(device)
    loss2_weight = 0.1

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
        #print("init_freq_mapping: ", init_freq_mapping)

        current_freq_mapping = torch.zeros([model.output_mapping.target_class_num, model.output_mapping.target_class_num*81]).to(device)
        for i in range(model.output_mapping.target_class_num):
            for j in model.output_mapping.self_definded_map[i]:
                current_freq_mapping[i, j] = 1.0
    
    
    f = open(fname, "a")
    best_result = [-1, 0., 0., 1.] # epoch, traing acc, validation acc, resize scale
    total_train_acc = 0
    total_valid_acc = 0
    scale_grad = []

    scaler = GradScaler()
    for epoch in range(Epoch):
        # Frequency mapping
        if(model.output_mapping.mapping_method == "frequency_based_mapping" and freqmap_interval != None and epoch!= 0 and epoch%freqmap_interval == 0):
            FreqLabelMap(model, trainloader, device, wild_dataset=wild_dataset)
            # print([model.text_content[x].item() for x in model.output_mapping.self_definded_map])
            freq_conv_loss_update = 1
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
                if(convergence == True):
                    if(model.output_mapping.mapping_method == "fully_connected_layer_mapping"):
                        loss2 = (MseLoss(model.output_mapping.layers.weight, init_weight) + MseLoss(model.output_mapping.layers.bias, zero_bias)) * loss2_weight 
                        loss += loss2
                    elif(model.output_mapping.mapping_method == "frequency_based_mapping" and freqmap_interval != None and freq_conv_loss_update == 1):
                        loss2 = MseLoss(current_freq_mapping, init_freq_mapping) * loss2_weight 
                        loss += loss2
                        freq_conv_loss_update = 0
            
            scaler.scale(loss).backward()

            # clip scale's gradient
            if(model.no_trainable_resize == 0): 
                nn.utils.clip_grad_value_(model.train_resize.scale, 0.001)


            if(model.output_mapping.mapping_method == "fully_connected_layer_mapping"):
                nn.utils.clip_grad_value_(model.output_mapping.layers.bias, 0.001) 
                nn.utils.clip_grad_value_(model.output_mapping.layers.weight, 0.001)
            

            scaler.step(optimizer)
            scaler.update()
            model.model.logit_scale.data = torch.clamp(model.model.logit_scale.data, 0, 4.6052) # Clamps all elements in input into the range in CLIP model 

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

        # update log
        if(model.no_trainable_resize == 0):
            f.write(f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}, ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Loss2: {total_train_loss2:.4f}, Scale: {model.train_resize.scale.item():.4f}\n")
        else:
            f.write(f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}, ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Loss2: {total_train_loss2:.4f}\n")
        
        if(epoch%10 ==0 or epoch == Epoch-1): 
            # Validation
            model.eval()
            valid_loss = []
            valid_accs = []
            pbar = tqdm(testloader, total=len(
                testloader), desc=f"Epoch {epoch+1} Testing", ncols=160)
            for pb in pbar:
                if(wild_dataset == True):
                    imgs, labels, _ = pb
                else:
                    imgs, labels = pb

                if imgs.get_device() == -1:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                with torch.no_grad():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()

                valid_loss.append(loss.item())
                valid_accs.append(acc)

                total_valid_loss = sum(valid_loss) / len(valid_loss)
                total_valid_acc = sum(valid_accs) / len(valid_accs)
                pbar.set_postfix_str(f"ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}")

            # update log
            f.write(f"Epoch {epoch+1} Testing, ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}\n")

            if (total_valid_acc > best_result[2] or epoch == Epoch - 1):
                ss = None
                if(model.no_trainable_resize == 0):
                    ss = model.train_resize.scale.item()
                # save model
                print("Save! Acc: ", total_valid_acc, ", Scale: ", ss)
                f.write(f"Save! Acc: {total_valid_acc}, Scale: {ss}\n")
                state_dict = {
                    "pretrained_model": model.model_name,
                    "resize_dict": model.train_resize.state_dict(),
                    "perturb_dict": model.input_perturbation.state_dict(),
                    "outmap_dict": model.output_mapping.state_dict(),
                    "output_mapping": model.output_mapping.self_definded_map,
                    "mapping_method": model.output_mapping.mapping_method,
                    "freq_check": model.output_mapping.freq_check,
                    "optimizer_dict": optimizer.state_dict(),
                    "init_scale" : model.init_scale
                }
                if(total_valid_acc > best_result[2]):
                    best_result = [epoch, total_train_acc, total_valid_acc, ss]
                    torch.save(state_dict, str(dataset) + "_best.pth")
                if(epoch == Epoch - 1):
                    torch.save(state_dict, str(dataset) + "_last.pth")
    f.close()

    return best_result

def Training_pure(fname, model, trainloader, testloader, Epoch, lr, device, FF=False, wild_dataset=False):   
    # Update stretagy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-5) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*Epoch), int(0.72*Epoch)], gamma=0.1)

    total_train_acc = 0
    total_valid_acc = 0
    best_result = [-1, 0., 0.] # epoch, traing acc, validation acc
    f = open(fname, "w+")
    scaler = GradScaler()
    for epoch in range(Epoch):
        # Training
        if(FF == True): # full finetune all the layers
            model.train()
        train_loss = []
        train_accs = []
        pbar = tqdm(trainloader, total=len(trainloader),
                    desc=f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            optimizer.zero_grad()
            with autocast():
                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

            total_train_loss = sum(train_loss) / len(train_loss)
            total_train_acc = sum(train_accs) / len(train_accs)
            pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}")
        # update log
        f.write(f"Epoch {epoch+1} Training, ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}\n")

        scheduler.step()

        if(epoch%10 ==0 or epoch == Epoch-1): 
            # Validation
            model.eval()
            valid_loss = []
            valid_accs = []
            pbar = tqdm(testloader, total=len(
                testloader), desc=f"Epoch {epoch+1} Testing", ncols=100)
            for pb in pbar:
                if(wild_dataset == True):
                    imgs, labels, _ = pb
                else:
                    imgs, labels = pb

                with torch.no_grad():
                    logits = model(imgs.to(device))
                    loss = criterion(logits, labels.to(device))
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                valid_loss.append(loss.item())
                valid_accs.append(acc)

                total_valid_loss = sum(valid_loss) / len(valid_loss)
                total_valid_acc = sum(valid_accs) / len(valid_accs)
                pbar.set_postfix_str(f"ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}")
            # update log
            f.write(f"Epoch {epoch+1} Testing, ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}\n")

            if total_valid_acc > best_result[2]:
                best_result = [epoch, total_train_acc, total_valid_acc]
    
    # save model
    print("Save! Acc: ", total_valid_acc)
    state_dict = {"model": model}
    torch.save(state_dict, fname.split(".")[0]+ ".pth")
    
    f.close()
    return best_result[2]

def Training_pure_clip(fname, model, trainloader, testloader, Epoch, lr, device, FF=False, wild_dataset=False):   
    # loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    t_max = Epoch * len(trainloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    total_train_acc = 0
    total_valid_acc = 0
    best_result = [-1, 0., 0.] # epoch, traing acc, validation acc
    f = open(fname, "w+")
    scaler = GradScaler()
    for epoch in range(Epoch):
        # Training
        if(FF == True): # full finetune all the layers
            model.train()
        train_loss = []
        train_accs = []
        pbar = tqdm(trainloader, total=len(trainloader),
                    desc=f"Epoch {epoch+1} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            optimizer.zero_grad()
            with autocast():
                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            nn.utils.clip_grad_value_(model.linear.bias, 0.001)
            nn.utils.clip_grad_value_(model.linear.weight, 0.001)

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

            total_train_loss = sum(train_loss) / len(train_loss)
            total_train_acc = sum(train_accs) / len(train_accs)
            pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}")
        # update log
        f.write(f"Epoch {epoch+1} Training, ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}\n")

        if(epoch%10 == 0 or epoch == Epoch-1): 
            # Validation
            model.eval()
            valid_loss = []
            valid_accs = []
            pbar = tqdm(testloader, total=len(
                testloader), desc=f"Epoch {epoch+1} Testing", ncols=100)
            for pb in pbar:
                if(wild_dataset == True):
                    imgs, labels, _ = pb
                else:
                    imgs, labels = pb

                with torch.no_grad():
                    logits = model(imgs.to(device))
                    loss = criterion(logits, labels.to(device))
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                valid_loss.append(loss.item())
                valid_accs.append(acc)

                total_valid_loss = sum(valid_loss) / len(valid_loss)
                total_valid_acc = sum(valid_accs) / len(valid_accs)
                pbar.set_postfix_str(f"ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}")
            # update log
            f.write(f"Epoch {epoch+1} Testing, ACC: {total_valid_acc*100:.2f}%, Loss: {total_valid_loss:.4f}\n")

            if total_valid_acc > best_result[2]:
                best_result = [epoch, total_train_acc, total_valid_acc]
    
    # save model
    print("Save! Acc: ", total_valid_acc)
    state_dict = {"model": model}
    torch.save(state_dict, fname.split(".")[0]+ ".pth")
    
    f.close()
    return best_result[2]

def LP(fname, model, pretrained_model, class_num, trainloader, testloader, Epoch, lr, device, wild_dataset=False):
    for param in model.parameters():
        param.requires_grad = False
    model.eval() 

    # Parameters of newly constructed modules have requires_grad=True by default
    if(pretrained_model == "vit_b_16"):
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, class_num)
    elif(pretrained_model == "swin_t"):
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, class_num)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)

    Trainable_Parameter_Size(model, fname)
    #'''
    params_to_update = []
    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    #'''
    best_val_acc = Training_pure(fname, model.to(device), trainloader, testloader, Epoch, lr, device, wild_dataset=wild_dataset)

    return best_val_acc

def Full_Finetune(fname, model, pretrained_model, class_num, trainloader, testloader, Epoch, lr, device, wild_dataset=False):
    # Parameters of newly constructed modules have requires_grad=True by default
    if(pretrained_model == "vit_b_16"):
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, class_num)
    elif(pretrained_model == "swin_t"):
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, class_num)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)

    Trainable_Parameter_Size(model, fname)
    #'''
    params_to_update = []
    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    #'''

    best_val_acc = Training_pure(fname, model.to(device), trainloader, testloader, Epoch, lr, device, FF=True, wild_dataset=wild_dataset)

    return best_val_acc

def get_saparate_text_embedding(classnames, templates, model, device): #####@@@@@@@
    zeroshot_weights = []
    if isinstance(templates, list):
        for template in tqdm(templates, desc="Embedding texts", ncols=100):
            texts = [template.format(classname) for classname in classnames]
            texts = clip.tokenize(texts).to(device)
            with torch.no_grad():
                text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(text_embeddings)
    else:
        texts = [templates.format(classname) for classname in classnames]
        texts = clip.tokenize(texts).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights = text_embeddings

    return zeroshot_weights

def CLIP_Text_Embedding(class_names, template_number, clip_model, device):
    TEMPLATES = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES # len(TEMPLATES): 81
    txt_emb = get_saparate_text_embedding(class_names, TEMPLATES[template_number], clip_model, device) 
    return txt_emb

# https://github.com/openai/CLIP
def CLIP_Pure(model, testloader, class_names, device, wild_dataset=False):
    model.requires_grad_(False)
    model.eval()
    # Prepare text embedding
    template_number = 0 # use default template
    txt_emb = CLIP_Text_Embedding(class_names, template_number, model, device) 

    total_train_acc = 0
    total_valid_acc = 0
    valid_accs = []
        
    # Validation
    pbar = tqdm(testloader, total=len(testloader), ncols=120)
    for pb in pbar:
        if(wild_dataset == True):
            imgs, labels, _ = pb
        else:
            imgs, labels = pb

        if imgs.get_device() == -1:
            imgs = imgs.to(device)
            labels = labels.to(device)

        with torch.no_grad():
            x_emb = model.encode_image(imgs)

        x_emb /= x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        valid_accs.append(acc)

        total_valid_acc = sum(valid_accs) / len(valid_accs)
        pbar.set_postfix_str(f"ACC: {total_valid_acc*100:.2f}%")

    return total_valid_acc

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, out_dim, clip_model):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        self.clip_model = clip_model

    def forward(self, x):
        x = self.clip_model.encode_image(x) # output shape: [128, 512]
        x = self.linear(x)
        return x

# ref: https://github.com/openai/CLIP
# use the image features to classify
def CLIP_LP(fname, model, trainloader, testloader, class_num, Epoch, lr, device, b_l="b", wild_dataset=False): 
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  
    if(b_l == "b"):
        LR_model = LogisticRegression(512, class_num, model) 
    else:
        LR_model = LogisticRegression(768, class_num, model) 
    Trainable_Parameter_Size(LR_model, fname)

    print("Params to learn:")
    for name,param in LR_model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    
    best_val_acc = Training_pure_clip(fname, LR_model.to(device), trainloader, testloader, Epoch, lr, device, wild_dataset=wild_dataset)
    return best_val_acc
