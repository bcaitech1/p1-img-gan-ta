import os
import sys
import argparse
import json
import logging
import random

import torch
import torch.nn as nn # 임시 테스트용
import torch.utils.data as data
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

from sklearn.model_selection import StratifiedKFold

import wandb

data_dir = '/opt/ml/input/data/train'
img_dir = f'{data_dir}/images'
info = pd.read_csv(f'{data_dir}/train.csv')


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Enter essential parameters')


class ModelExistError(Exception):
    def __init__(self):
        super().__init__('model not exist')

class ModelSuitableError(Exception):
    def __init__(self):
        super().__init__('model not suitable')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#def seed_everything(seed: int = 1930):
#
#    print("Using Seed Number {}".format(seed))
#
#    os.environ["PYTHONHASHSEED"] = str(
#        seed)  # set PYTHONHASHSEED env var at fixed value
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
#    np.random.seed(seed)  # for numpy pseudo-random generator
#    random.seed(
#        seed)  # set fixed value for python built-in pseudo-random generator
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    torch.backends.cudnn.enabled = False


def get_model(model_name):
    """
    get model object

    Args:
        model_name : model name

    Returns:
        model object
    """
    if model_name == 'resnext':
        return models.MyResNext()
    elif model_name == 'effi':
        return models.MyEfficentNetb4()
    elif model_name == 'multi_resnext':
        return models.MyMultiResNext()
    else:
        raise ModelExistError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    parser.add_argument('-s', '--save', default=None, type=str,
                        help='save path')

    args = parser.parse_args()

    if args.config == None or args.save == None:
        raise ParameterError

    # get hyperparam
    with open(args.config) as json_file:
        json_data = json.load(json_file)

    # setting args
    batch_size = json_data["batch_size"]
    epochs = json_data["epochs"]
    learning_rate = json_data["learning_rate"]
    weight_decay = json_data["weight_decay"]
    loss_gamma = json_data["loss_gamma"]
    seed = json_data["seed"]
    model_name = json_data["model"]
    transform_type = json_data["transform_type"]
    mixup = bool(json_data["mixup"])
    cutmix = bool(json_data["cutmix"])
    upsample = bool(json_data["upsample"])
    extend_sample = bool(json_data["extend_sample"])
    train_type = json_data["train_type"]

    print("*******************SETTING SET*******************")
    print("batch_size : ", batch_size)
    print("epochs : ", epochs)
    print("learning rate : ", learning_rate)
    print("weight_decay : ", weight_decay)
    print("loss_gamma : ", loss_gamma)
    print("seed : ", seed)
    print("model name : ", model_name)
    print("transform type : ", transform_type)
    print("mixup : ", mixup)
    print("cutmix : ", cutmix)
    print("upsample : ", upsample)
    print("extend sample : ", extend_sample)
    print("train type : ", train_type)
    print("**************************************************")
    
    # setting seed
    seed_everything(seed =seed)

    PATH = args.save

    # setting path
    sys.path.append('/opt/ml/pstage01')
    from model import models, loss, metric
    from dataloader import mask
    from util import meter, transformers
    import fold, novalid, multi, normal

    # setting device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    use_cuda = torch.cuda.is_available()

    # RGB mean, std
    mean, std = (0.56019358, 0.52410121, 0.501457), (0.23318603, 0.24300033, 0.24567522)

    transforms = transformers.get_transforms(mean=mean, std=std, transform_type=transform_type)

    if train_type == 'normal':
        dataset = mask.MaskBaseDataset(
            data_dir= data_dir,
            img_dir=img_dir,
            upsample=upsample,
            extend_sample=extend_sample
        )

        # split train, valid(9:1)
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

        train_dataset.dataset.set_transform(transforms['train'])
        val_dataset.dataset.set_transform(transforms['val'])

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory = use_cuda,
            drop_last = True
        )

        valid_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory = use_cuda,
            drop_last = True
        )
        
#        images, labels, _ = next(iter(train_loader))

        wandb.login() # wandb

        wandb_config = dict(
            epochs = epochs,
            batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            loss_gamma = loss_gamma,
            seed = seed,
            model_name = model_name,
            transform_type = transform_type,
            mixup = mixup,
            cutmix = cutmix
        )

        with wandb.init(project=PATH.split("/")[-1], config=wandb_config):
            model = get_model(model_name)
            model.to(device)
            torch.nn.DataParallel(model)

            criterion = loss.FocalLoss(gamma=loss_gamma)

            optimizer = Adam(model.parameters(), lr=learning_rate)
            # optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay= weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

            wandb.watch(model, criterion, log="all", log_freq=10) # wandb 어떤거 조사시킬지 설정

            normal.train(
                model = model, 
                epochs = epochs, 
                train_loader = train_loader, 
                valid_loader = valid_loader, 
                save_path = PATH,
                criterion= criterion,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device,
                wandb = wandb,
                mixup = mixup, 
                cutmix = cutmix
                )

            wandb.save("model.onnx") # 정보 save
            model.cpu()

    elif train_type == 'fold':
        def get_mask_label(image_name):
            if 'incorrect_mask' in image_name:
                return 1
            elif 'normal' in image_name:
                return 2
            elif 'mask' in image_name:
                return 0
            else:
                raise ValueError(f'No class for {image_name}')

        def get_gender_label(gender):
            return 0 if gender == 'male' else 1

        def get_age_label(age):
            return 0 if int(age) < 30 else 1 if int(age) < 60 else 2
        
        def convert_gender_age(gender, age):
            gender_label = get_gender_label(gender)
            age_label = get_age_label(age)
            return gender_label * 3 + age_label

        info['gender_age'] = info.apply(lambda x: convert_gender_age(x.gender, x.age), axis=1)

        skf = StratifiedKFold(n_splits=5 , shuffle=True)
        info.loc[:, 'fold'] = 0
        for fold_num, (train_index, val_index) in enumerate(skf.split(X=info.index, y=info.gender_age.values)):
            info.loc[info.iloc[val_index].index, 'fold'] = fold_num

        image_dir = os.path.join(data_dir, 'images')

        for fold_idx in range(5):
            train = info[info.fold != fold_idx].reset_index(drop=True)
            val = info[info.fold == fold_idx].reset_index(drop=True)
            
            train_dataset = mask.MaskDatasetFold(image_dir, train, transforms['train'])
            val_dataset = mask.MaskDatasetFold(image_dir, val, transforms['val'])
            
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory = use_cuda,
                drop_last = True
            )

            valid_loader = data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory = use_cuda,
                drop_last = True
            )

            model = get_model(model_name)
            model.cuda()

            criterion = loss.FocalLoss(gamma=loss_gamma)

            optimizer = Adam(model.parameters(), lr=learning_rate)
            # optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay= weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

            fold.train(
                fold_count=fold_idx,
                model=model, 
                epochs=epochs, 
                train_loader=train_loader, 
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device, 
                save_path=PATH)
            model.cpu()
    
    elif train_type == "novalid":
        
        train_dataset = mask.MaskBaseDataset(
            data_dir= data_dir,
            img_dir=img_dir,
            upsample=upsample,
            extend_sample=extend_sample
        )

        train_dataset.set_transform(transforms['train'])

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory = use_cuda,
            drop_last = True
        )

        model = get_model(model_name)
        model.cuda()

        criterion = loss.FocalLoss(gamma=loss_gamma)

        optimizer = Adam(model.parameters(), lr=learning_rate)
        # optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay= weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

        novalid.train(
            model=model, 
            epochs=epochs, 
            train_loader=train_loader, 
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device, 
            save_path=PATH
        )
        model.cpu()

    elif train_type == "multi":
        if "multi" not in model_name:
            raise ModelSuitableError


        dataset = mask.MultiHeadMaskDataset(
            data_dir= data_dir,
            img_dir=img_dir,
            upsample=upsample,
            extend_sample=extend_sample
            )
        
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

        train_dataset.dataset.set_transform(transforms['train'])
        val_dataset.dataset.set_transform(transforms['val'])

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory = use_cuda,
            drop_last = True
        )

        valid_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory = use_cuda,
            drop_last = True
        )

        wandb.login() # wandb

        wandb_config = dict(
            epochs = epochs,
            batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            loss_gamma = loss_gamma,
            seed = seed,
            model_name = model_name,
            transform_type = transform_type,
            mixup = mixup,
            cutmix = cutmix
        )

        with wandb.init(project=PATH.split("/")[-1], config=wandb_config):
            model = get_model(model_name)
            model.cuda()

            criterion = loss.FocalLoss(gamma=loss_gamma)

            optimizer = Adam(model.parameters(), lr=learning_rate)
            # optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay= weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

            wandb.watch(model, criterion, log="all", log_freq=10) # wandb 어떤거 조사시킬지 설정

            multi.train(
                model = model, 
                epochs = epochs, 
                train_loader = train_loader, 
                valid_loader = valid_loader, 
                criterion = criterion,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device,
                save_path = PATH,
                wandb = wandb
                )
            wandb.save("model.onnx") # 정보 save
            model.cpu()
