# This code is based on https://github.com/OPTML-Group/ILM-VP. ILM_Dataloader and some datasets are downloaded from ILM-VP.
import auto_vp.datasets as datasets
from auto_vp.const import GTSRB_LABEL_MAP
from auto_vp.ILM_Dataloader import COOPLMDBDataset

from torch.utils.data import DataLoader
import torchvision
import torch
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm.auto import tqdm
import os

def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def DataPrepare(dataset_name, dataset_dir, target_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), download=True, batch_size=64, random_state=1, clip_transform=None, CIFAR10_C_mode="gaussian_noise"):
    if(clip_transform == None):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(target_size)])
            # torchvision.transforms.Normalize(mean, std)
    else:
        transform = clip_transform

    if dataset_name == "ABIDE":
        trainset = datasets.ABIDE(download=download, data_path=dataset_dir,
                                  mode='train', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.ABIDE(download=download, data_path=dataset_dir,
                                 mode='test', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        class_names = ["Autism", "Control"]

    elif dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        class_names = refine_classnames(testset.classes)

    elif dataset_name == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        class_names = refine_classnames(testset.classes)

    elif dataset_name == "Melanoma":            
        trainset = datasets.Melanoma(download=download, data_path=dataset_dir,
                                     mode='train', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state, transformer=transform)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.Melanoma(download=download, data_path=dataset_dir,
                                    mode='test', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state, transformer=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    
    elif dataset_name == "SVHN":
        trainset = torchvision.datasets.SVHN(root=dataset_dir, split="train",
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.SVHN(root=dataset_dir,  split="test",
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        class_names = [f'{i}' for i in range(10)]

    elif dataset_name == "GTSRB":
        trainset = torchvision.datasets.GTSRB(root=dataset_dir, split="train",
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.GTSRB(root=dataset_dir,  split="test",
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))

    elif dataset_name == "Flowers102":
        trainset = COOPLMDBDataset(root=dataset_dir, split="train", transform = transform)
        testset = COOPLMDBDataset(root=dataset_dir, split="test", transform = transform)
        class_names = refine_classnames(testset.classes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset_name == "DTD":
        trainset = COOPLMDBDataset(root=dataset_dir, split="train", transform = transform)
        testset = COOPLMDBDataset(root=dataset_dir, split="test", transform = transform)
        class_names = refine_classnames(testset.classes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset_name == "Food101":
        trainset = torchvision.datasets.Food101(root=dataset_dir, split="train",
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.Food101(root=dataset_dir,  split="test",
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        class_names = refine_classnames(testset.classes)

    elif dataset_name == "EuroSAT":
        trainset = COOPLMDBDataset(root=dataset_dir, split="train", transform = transform)
        testset = COOPLMDBDataset(root=dataset_dir, split="test", transform = transform)
        class_names = refine_classnames(testset.classes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset_name == "OxfordIIITPet":
        trainset = torchvision.datasets.OxfordIIITPet(root=dataset_dir, split="trainval",
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.OxfordIIITPet(root=dataset_dir,  split="test",
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        class_names = refine_classnames(testset.classes)

    elif dataset_name == "UCF101":        
        trainset = COOPLMDBDataset(root=dataset_dir, split="train", transform = transform)
        testset = COOPLMDBDataset(root=dataset_dir, split="test", transform = transform)
        class_names = refine_classnames(testset.classes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset_name == "CIFAR10-C":        
        trainset = None
        trainloader = None

        testset = datasets.CIFAR10_C(download=download, data_path=dataset_dir, mode=CIFAR10_C_mode, target_size=target_size, transformer=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        class_names = testset.classes

    elif dataset_name == "Camelyon17": # https://wilds.stanford.edu/get_started/
        dataset = get_dataset(dataset="camelyon17", download=download)

        trainset = dataset.get_subset("train", transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                    shuffle=True, num_workers=2)

        testset = dataset.get_subset("test", transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        class_names = ["normal", "tumor"]

    elif dataset_name == "Iwildcam": # https://wilds.stanford.edu/get_started/
        dataset = get_dataset(dataset="iwildcam", download=download)

        trainset = dataset.get_subset("train", transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

        testset = dataset.get_subset("test", transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        #category_path = "./data/iwildcam_v2.0/categories.csv"
        category_path = os.path.join(dataset_dir, "/data/iwildcam_v2.0/categories.csv")

        df = pd.read_csv(category_path)
        class_names = df['name'].head(182).values.tolist()
        class_names.append('animal')

    elif dataset_name == "FMoW": # https://wilds.stanford.edu/get_started/
        dataset = get_dataset(dataset="fmow", download=download)

        trainset = dataset.get_subset("train", transform=transform)
        trainloader =torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                    shuffle=True, num_workers=2)

        testset = dataset.get_subset("test", transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

        # ref: https://github.com/fMoW/dataset
        class_names = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]

    elif dataset_name == "Spawrious":
        dirpath = "spawrious224/0"
        trainset = datasets.Spawrious(download=download, data_path=os.path.join(dataset_dir,dirpath),
                                     mode='train', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state, transformer=transform)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.Spawrious(download=download, data_path=os.path.join(dataset_dir,dirpath),
                                    mode='test', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state, transformer=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        class_names = ["bulldog", "corgi", "dachshund", "labrador"]

    elif dataset_name == "ImageNet1k":
        if(clip_transform == None):
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                torchvision.transforms.Resize(target_size),
                torchvision.transforms.ToTensor()])
        else:
            transform = clip_transform

        trainset = datasets.ImageNet1k(download=download, data_path=dataset_dir,
                                     mode='train', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state, transformer=transform)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.ImageNet1k(download=download, data_path=dataset_dir,
                                    mode='test', target_size=target_size, data_mean=mean, data_std=std, random_state=random_state, transformer=transform)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        class_names = refine_classnames(trainset.classes)
    else:
        raise NotImplementedError(f"{dataset_name} not supported")

    return trainloader, testloader, class_names, trainset

def Data_Scalability(trainset, scalibility_rio, batch_size, mode="random", random_state=1, wild_dataset=False):
    total_index = range(0,len(trainset))
    if(mode == "random"):
        kf = KFold(n_splits=scalibility_rio, shuffle=True, random_state=random_state)
        for (big_ids, small_ids) in kf.split(total_index):
            print(small_ids)
            break
        small_subsampler = torch.utils.data.SubsetRandomSampler(small_ids)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=small_subsampler)
    elif(mode == "equal"):
        targets = []
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        pbar = tqdm(trainloader, total=len(trainloader))
        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb
            targets += labels
        print(len(targets))

        big_ids, small_ids = train_test_split(total_index, test_size=1/scalibility_rio, random_state=random_state, shuffle=True, stratify=targets)
        print(len(small_ids))

        small_subsampler = torch.utils.data.SubsetRandomSampler(small_ids)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=small_subsampler)
        #### Check the Spilt ####
        count = {}
        pbar = tqdm(trainloader, total=len(trainloader))
        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            for lab in labels:
                if lab.item() in count:
                    count[lab.item()] =  count[lab.item()] + 1
                else:
                    count[lab.item()] = 1
        print(count) 
    else:
        raise NotImplementedError(f"{mode} not supported")
    return trainloader
