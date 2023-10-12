import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import clip
from tqdm.auto import tqdm
import numpy as np
import timm

from .const import DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES

# ref: https://github.com/RobustBench/robustbench/blob/master/robustbench/utils.py
# ref: https://pytorch.org/vision/0.8/models.html

class BaseWrapper(nn.Module):
    def __init__(self, model_name=None, dataset_name=None, input_perturbation=None, output_mapping=None, train_resize=None, init_scale=1.0, clip_img_size=128, device=None):
        super(BaseWrapper, self).__init__()
        self.model = None
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.No_operation = nn.Identity()  # do nothing, just forwarding the input
        self.no_pretrained_model = 0
        self.no_trainable_resize = 0
        self.no_output_mapping = 0
        self.device = device
        self.clip_preprocess = None
        self.text_content = []
        self.init_scale = init_scale
        self.clip_rz_transform = transforms.Resize([clip_img_size, clip_img_size])

        if(model_name == None):
            self.model = self.No_operation.to(device)
            self.no_pretrained_model = 1
            
        if(train_resize == None):
            self.train_resize = self.No_operation.to(device)
            self.no_trainable_resize = 1
        else:
            self.train_resize = train_resize.to(device)

        if(input_perturbation == None):
            self.input_perturbation = self.No_operation.to(device)
        else:
            self.input_perturbation = input_perturbation.to(device)

        if(output_mapping == None):
            self.output_mapping = self.No_operation.to(device)
            self.no_output_mapping = 1
        else:
            self.output_mapping = output_mapping.to(device)
        
        # Initial output_mapping.layers weight
        if((self.model_name == "clip" or self.model_name == "clip_large") and self.output_mapping.mapping_method == "fully_connected_layer_mapping" and self.output_mapping.weightinit == True):
            w = torch.zeros([self.output_mapping.target_class_num, self.output_mapping.target_class_num*81])
            for i in range(self.output_mapping.target_class_num):
                w[i, i] = 1.0
            with torch.no_grad():
                self.output_mapping.layers.weight.copy_(w)

        self.model_zoo = ["vgg16_bn", "resnet18", "resnet50", "resnext101_32x8d", "ig_resnext101_32x8d", "vit_b_16", "clip", "clip_large", "clip_ViT_B_32", "swin_t"]

        # load model from model zoo
        if self.model_name in self.model_zoo:
            if self.model_name == "vgg16_bn": # VGG-16 with batch normalization
                model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).to(device)  
            elif self.model_name == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
            elif self.model_name == "resnet50":
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
            elif self.model_name == "resnext101_32x8d":
                model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT).to(device)
            elif self.model_name == "ig_resnext101_32x8d": # https://paperswithcode.com/model/ig-resnext?variant=ig-resnext101-32x8d
                model = timm.create_model('ig_resnext101_32x8d', pretrained=True).to(device)
            elif self.model_name == "vit_b_16":
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)
            elif self.model_name == "swin_t":
                model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT).to(device)
            elif self.model_name == "clip" or self.model_name == "clip_ViT_B_32":
                model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                # https://github.com/openai/CLIP/issues/57
                for p in model.parameters(): 
                    p.data = p.data.float() 
                    if p.grad:
                        p.grad.data = p.grad.data.float()
            elif self.model_name == "clip_large":
                model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
                # https://github.com/openai/CLIP/issues/57
                for p in model.parameters(): 
                    p.data = p.data.float() 
                    if p.grad:
                        p.grad.data = p.grad.data.float()
            

            # Frozen the pretrained model
            model.requires_grad_(False)

            # Set to evaluation mode
            self.model = model.eval()


    # ref : https://github.com/OPTML-Group/ILM-VP
    def get_saparate_text_embedding(self, classnames, templates, model):
        zeroshot_weights = []
        if isinstance(templates, list):
            for template in tqdm(templates, desc="Embedding texts", ncols=100):
                texts = [template.format(classname) for classname in classnames]
                self.text_content.append(texts)
                texts = clip.tokenize(texts).to(self.device)
                with torch.no_grad():
                    text_embeddings = model.encode_text(texts)
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                zeroshot_weights.append(text_embeddings)
            self.text_content = np.concatenate(self.text_content)
        else:
            texts = [templates.format(classname) for classname in classnames]
            self.text_content = texts
            texts = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights = text_embeddings

        return zeroshot_weights

    def CLIP_Text_Embedding(self, class_names, template_number):
        TEMPLATES = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES # len(TEMPLATES): 81
        self.txt_emb = torch.cat(self.get_saparate_text_embedding(class_names, TEMPLATES, self.model)) 
        return
    
    def CLIP_network(self, x):
        if(self.txt_emb == None):
            print("Error: no text embedding exist")
            return
        x_emb = self.model.encode_image(x)
        x_emb /= x_emb.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * x_emb @ self.txt_emb.t()
        return logits


    def forward(self, input):
        # clip need to resize by ourself 
        x = self.clip_rz_transform(input)

        img_h = -1
        img_w = -1
        if(self.no_trainable_resize == 0):
            x, img_h, img_w = self.train_resize(x)
        else:
            x = self.train_resize(x)

        x = self.input_perturbation(x, img_h, img_w)

        if(self.model_name == "clip_ViT_B_32"):
            x = self.model.encode_image(x)
        elif(self.model_name[0:4] == "clip"):
            x = self.CLIP_network(x)
        else:
            x = self.model(x)
        x = self.output_mapping(x)
        return x
