import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import kornia
import numpy as np
import matplotlib.pyplot as plt
import clip
from tqdm.auto import tqdm
# ref: https://github.com/Prinsphield/Adversarial_Reprogramming/blob/master/main.py


class Trainable_Resize(nn.Module):
    def __init__(self, output_size=(3, 256, 256)):
        super(Trainable_Resize, self).__init__()
        self.height_out = output_size[1]
        self.width_out = output_size[2]
        self.scale = Parameter(torch.tensor(1.0), requires_grad=True)
        self.l_pad = 0.0  # left padding
        self.r_pad = 0.0  # right padding
        self.u_pad = 0.0  # upper padding
        self.d_pad = 0.0  # lower padding

    def forward(self, image):   # image.shape [batch_sz, channel, H, W]
        self.l_pad = int((self.width_out-image.shape[3]+1)/2)
        self.r_pad = int((self.width_out-image.shape[3])/2)
        self.u_pad = int((self.height_out-image.shape[2]+1)/2)
        self.d_pad = int((self.height_out-image.shape[2])/2)

        x = torch.nn.functional.pad(
            image, (self.l_pad, self.r_pad, self.u_pad, self.d_pad), value=0)
        x = kornia.geometry.transform.scale(x, self.scale)

        return x, min(int(image.shape[2]*self.scale), self.height_out), min(int(image.shape[3]*self.scale), self.width_out)


class InputPadding(nn.Module):
    def __init__(self, img_size=(3, 32, 32), output_size=(3, 256, 256), normalization=None, input_aware=False, padding_size=None, model_name=None, dataset_name=None, device=None):
        super(InputPadding, self).__init__()
        self.img_size = img_size
        self.channel = img_size[0]
        self.img_h = img_size[1]
        self.img_w = img_size[2]

        self.output_size = output_size
        self.out_h = output_size[1]
        self.out_w = output_size[2]

        self.u_pad = int((output_size[1]-img_size[1]+1)/2)
        self.d_pad = int((output_size[1]-img_size[1])/2)
        self.l_pad = int((output_size[2]-img_size[2]+1)/2)
        self.r_pad = int((output_size[2]-img_size[2])/2)

        self.normalization = normalization

        self.padding_size = padding_size

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.device = device

        self.input_aware = input_aware

        self.mask = None
        self.init_mask()

        self.delta = torch.nn.Parameter(
            data=torch.zeros(3, output_size[1], output_size[2]))
        self.dropout = nn.Dropout(0.2)

    def init_mask(self):
        self.mask = torch.ones(self.output_size).to(self.device)
        # upper triangle and the diagonal are set to True, others are False
        up_tri = np.invert(np.tri(N=self.img_h, M=self.img_w, k=0, dtype=bool))
        up_tri = torch.from_numpy(up_tri)
        if(self.padding_size == None or self.padding_size < int((self.out_h-self.img_h)//2)): 
            if(self.dataset_name == "ABIDE"):
                self.mask[:, int((self.out_h-self.img_h)//2):int((self.out_h+self.img_h)//2), int(
                    (self.out_w-self.img_w)//2):int((self.out_w+self.img_w)//2)] = torch.where(up_tri, 0, 1).to(self.device)
            else:
                self.mask[:, int((self.out_h-self.img_h)//2):int((self.out_h+self.img_h)//2), int(
                    (self.out_w-self.img_w)//2):int((self.out_w+self.img_w)//2)] = 0 # the location of img is set to zero
        else:
            self.mask[:, self.padding_size:self.out_h-self.padding_size, self.padding_size:self.out_w-self.padding_size] = 0 

    def redefind_mask(self, img, img_h, img_w):
        self.channel = img.size()[1]
        self.img_h = img.size()[2]
        self.img_w = img.size()[3]

        # replace with real size from Trainable_Resize
        if(img_h != -1):
            self.img_h = img_h
        if(img_w != -1):
            self.img_w = img_w

        self.mask = torch.ones(self.output_size).to(self.device)
        # upper triangle and the diagonal are set to True, others are False
        up_tri = np.invert(np.tri(N=self.img_h, M=self.img_w, k=0, dtype=bool))
        up_tri = torch.from_numpy(up_tri).to(self.device)
        if(self.padding_size == None or self.padding_size < int((self.out_h-self.img_h)//2)):
            if(self.dataset_name == "ABIDE"):
                self.mask[:, int((self.out_h-self.img_h)//2):int((self.out_h+self.img_h)//2), int(
                    (self.out_w-self.img_w)//2):int((self.out_w+self.img_w)//2)] = torch.where(up_tri, 0, 1).to(self.device)
            else:
                self.mask[:, int((self.out_h-self.img_h)//2):int((self.out_h+self.img_h)//2), int(
                    (self.out_w-self.img_w)//2):int((self.out_w+self.img_w)//2)] = 0 # the location of img is set to zero
        else:
            self.mask[:, self.padding_size:self.out_h-self.padding_size, self.padding_size:self.out_w-self.padding_size] = 0 

        self.u_pad = int((self.out_h-img.size()[2]+1)/2)
        self.d_pad = int((self.out_h-img.size()[2])/2)
        self.l_pad = int((self.out_w-img.size()[3]+1)/2)
        self.r_pad = int((self.out_w-img.size()[3])/2)

    def forward(self, image, img_h, img_w):
        self.redefind_mask(image, img_h, img_w)
        # 3-self.channel+1: times of channel repeat to fit the image net model
        image = image.repeat(1, 3-self.channel+1, 1, 1)

        x = torch.nn.functional.pad(
            image, (self.l_pad, self.r_pad, self.u_pad, self.d_pad), value=0)
        
        if(self.input_aware == True):
            print("Not Implement!")
            return

        if(self.model_name[0:4] == "clip"): 
            masked_delta = self.delta * self.mask
        else:
            masked_delta = torch.sigmoid(self.delta) * self.mask
        x_adv = x + masked_delta
        if (self.normalization != None):
            x_adv = self.normalization(x_adv)
        return x_adv


# Demo: https://colab.research.google.com/drive/1ZLUU4IafVv0oi_7ausP8A-XyO9ZZsEIJ#scrollTo=0nd6OhMZIgje
class Output_Mapping(nn.Module):
    def __init__(self, source_class_num=None, target_class_num=None, mapping_method=None, self_definded_map=None, num_source_to_map=None, weightinit=True, device=None):
        super(Output_Mapping, self).__init__()
        self.source_class_num = source_class_num
        self.target_class_num = target_class_num

        # mapping method: self_definded_mapping/random_mapping/frequency_based_mapping/fully_connected_layer_mapping
        self.mapping_method = mapping_method

        # [[source_i1], [source_i2], ....], [source_i1] map to target1
        self.self_definded_map = self_definded_map
        self.num_source_to_map = num_source_to_map

        self.weightinit = weightinit

        self.device = device

        # self.freq_ for frequency_based_mapping
        self.freq_check = False
        self.freq_is_map = [0 for _ in range(source_class_num)]

        # Semantic mapping
        self.sem_check = False
        self.layers = nn.Linear(self.source_class_num, self.target_class_num)


    def mapping_done(self):
        done = True
        for i in range(self.target_class_num):
            if(len(self.self_definded_map[i]) != self.num_source_to_map):
                done = False
                break
        return done

    # Frequency_ for frequency_based_mapping
    def Frequency_zero_padding(self, input, original_size=(32, 32), target_size=(64, 64)):
        if len(original_size) != 2 or len(target_size) != 2:
            print("Frequency_zero_padding: size dimension error.")
            return

        if original_size[0] != original_size[1] or target_size[0] != target_size[1]:
            print("Frequency_zero_padding: size must be square.")
            return

        if target_size[0] < original_size[0]:
            print("Frequency_zero_padding: target_size must larger than original_size.")
            return

        if (target_size[0] - original_size[0]) % 2 != 0:
            print("Frequency_zero_padding: size difference cannot divide by 2.")
            return

        padding_size = (target_size[0] - original_size[0])/2
        m = nn.ZeroPad2d(int(padding_size))
        return m(input)

    def Frequency_mapping(self, model, trainloader, device, wild_dataset=False):
        self.freq_is_map = [0 for _ in range(self.source_class_num)]
        preds, labs = self.Frequency_distribution_calculate(model, trainloader, device, wild_dataset)
        self.Frequency_mapping_define(preds, labs)
        if(self.mapping_done() == True):
            self.freq_check = True
        return

    def Frequency_distribution_calculate(self, model, trainloader, device, wild_dataset=False):
        model.eval()
        model.model.eval()
        labs = []
        preds = []
        pbar = tqdm(trainloader, total=len(trainloader))
        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            if imgs.get_device() == -1:
                imgs = imgs.to(device)
                labels = labels.to(device)

            with torch.no_grad():
                img_h = -1
                img_w = -1
                if(model.no_trainable_resize == 0):
                    x, img_h, img_w = model.train_resize(imgs)
                else:
                    x = model.train_resize(imgs)

                x = model.input_perturbation(x, img_h, img_w)

                if(model.model_name == "clip_ViT_B_32"):
                    x = model.model.encode_image(x)
                elif(model.model_name[0:4] == "clip"):
                    x = model.CLIP_network(x)
                else:
                    x = model.model(x)

            preds.append(x.argmax(dim=-1))
            labs.append(labels)

        preds = torch.cat(preds).cpu().int()
        labs = torch.cat(labs).cpu().int()
        return preds, labs

    def Frequency_mapping_define(self, preds, labs):
        self.self_definded_map = [[] for x in range(self.target_class_num)]
        mapped_matrix = np.zeros(self.source_class_num*self.target_class_num)
        for i in range(len(preds)):
            mapped_matrix[preds[i]*self.target_class_num + labs[i]] += 1
        
        i = -1
        while(self.mapping_done() == False and abs(i)<=len(mapped_matrix)):
            loc = mapped_matrix.argmax().item()
            mapped_matrix[loc] = -1
            m_target = loc % self.target_class_num 
            m_source = loc // self.target_class_num 
            if(len(self.self_definded_map[m_target]) < self.num_source_to_map and self.freq_is_map[m_source] != 1):
                self.self_definded_map[m_target].append(m_source)
                self.freq_is_map[m_source] = 1  # set is mapped 
            i -= 1
    
        while(self.mapping_done() == False):
            k = 0
            for i in range(self.target_class_num):
                for j in range(self.num_source_to_map - len(self.self_definded_map[i])):
                    while(k < self.source_class_num and self.freq_is_map[k] == 1):
                        k += 1
                    self.self_definded_map[i].append(k)
                    self.freq_is_map[k] = 1
                    k += 1
            if(self.mapping_done() == False):
                print("Error: not enough mapping source label")
                return

        return

    def get_saparate_text_embedding(self, classnames, model): 
        zeroshot_weights = []
        texts = clip.tokenize(classnames).to(self.device)
        with torch.no_grad():
            text_embeddings = model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights = text_embeddings
        return zeroshot_weights

    def Semantic_mapping(self, source_labels, target_labels, show_map = True):
        # load CLIP model
        CLIP_model, _ = clip.load("ViT-B/32", device=self.device)
        source_label_emb = self.get_saparate_text_embedding(source_labels, CLIP_model).cpu().numpy() 
        target_label_emb = self.get_saparate_text_embedding(target_labels, CLIP_model).cpu().numpy() 
        del CLIP_model # no further use

        # Calculate the cosine simularity
        # ref: https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
        simularity = np.dot(target_label_emb, source_label_emb.T) # (10, 1000)
        p1 = np.sqrt(np.sum(target_label_emb**2,axis=1))[:,np.newaxis]
        p2 = np.sqrt(np.sum(source_label_emb**2,axis=1))[np.newaxis,:]
        simularity = simularity/(p1*p2)

        # Define the mapping
        self.self_definded_map = [[] for x in range(self.target_class_num)]
        sort_id_sim = np.argsort(-simularity, axis=1) # "-" for descending order

        for k1 in range(self.num_source_to_map): # round robin mapping
            for i in range(self.target_class_num):
                idx = 0
                while(idx < self.source_class_num):
                    if(self.freq_is_map[sort_id_sim[i, idx]] == 0):
                        self.freq_is_map[sort_id_sim[i, idx]] = 1
                        self.self_definded_map[i].append(sort_id_sim[i, idx])
                        break
                    idx += 1
        
        # Check Whether the mapping is done
        if(self.mapping_done() == False):
            print("Error: not enough mapping source label")
            return
        else:
            self.sem_check = True
            if(show_map == True):
                print("Target Class:")
                print(target_labels)
                print("Semantic Map:")
                print([[source_labels[idx] for idx in self.self_definded_map[i]] for i in range(self.target_class_num)])
        return

    def forward(self, input):
        if self.mapping_method == "frequency_based_mapping" and self.freq_check == False:
            print("Error: no mapping exist")
            return
        if self.mapping_method == "semantic_mapping" and self.sem_check == False:
            print("Error: no mapping exist")
            return
        if self.mapping_method == "self_definded_mapping" or self.mapping_method == "frequency_based_mapping" or self.mapping_method == "semantic_mapping":
            output = [torch.mean(input[:, self.self_definded_map[x]], 1)
                      for x in range(len(self.self_definded_map))]
            output = torch.stack(output)
            output = torch.transpose(output, 1, 0)
        elif self.mapping_method == "fully_connected_layer_mapping":
            output = self.layers(input)

        return output
