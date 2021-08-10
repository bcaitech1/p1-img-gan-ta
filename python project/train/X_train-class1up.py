import os
import sys
import argparse
import json
import logging
import random

import torch
import torch.utils.data as data
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

data_dir = '/opt/ml/input/data/train'
img_dir = f'{data_dir}/images'


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Enter essential parameters')


class ModelExistError(Exception):
    def __init__(self):
        super().__init__('model not exist')


def seed_all(seed: int = 1930):
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(
        seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train(model, epochs, train_loader, valid_loader, save_path):
    best_valid_acc = 0
    best_valid_loss = 100000
    writer = SummaryWriter(log_dir=save_path + f"runs")  # write each model

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = meter.AverageMeter(), meter.AverageMeter()
        for iter, (img, label) in enumerate(tqdm(train_loader)):
            label_check = label.detach().cpu().numpy()
            if label_check[0] == 1:
                sampling_num = 2
            else:
                sampling_num = 1

            for i in range(sampling_num):
                optimizer.zero_grad()

                img, label = img.type(torch.FloatTensor).to(device), label.to(device)

                # 모델에 이미지 forward
                model.train()
                pred_logit = model(img)

                # loss 값 계산
                loss = criterion(pred_logit, label)

                # Backpropagation
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Accuracy 계산
                pred_label = torch.max(pred_logit.data, 1)

                # pred_label  = pred_logit.argmax(1)
                acc = (pred_label.indices == label).sum().item() / len(label)

                epoch_train_loss.update(loss.item(), len(img))
                epoch_train_acc.update(acc, len(img))

        #             if (iter %10 == 0) or (iter == len(train_loader)-1):
        #                 print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f" %(iter, len(train_loader), train_loss, train_acc))

        epoch_train_loss = epoch_train_loss.avg
        epoch_train_acc = epoch_train_acc.avg
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Acc/train', epoch_train_acc, epoch)
        print("Epoch %d | Train Loss %.4f | Train Acc %.4f" % (epoch, epoch_train_loss, epoch_train_acc))

        valid_loss, valid_acc, valid_f1 = meter.AverageMeter(), meter.AverageMeter(), meter.AverageMeter()
        model.eval()

        for img, label in valid_loader:
            img, label = img.type(torch.FloatTensor).to(device), label.to(device)

            with torch.no_grad():
                pred_logit = model(img)
            loss = criterion(pred_logit, label)
            pred_label = torch.max(pred_logit.data, 1)

            acc = (pred_label.indices == label).sum().item() / len(label)

            valid_loss.update(loss.item(), len(img))
            valid_acc.update(acc, len(img))
            valid_f1.update(metric.f1_loss(label, pred_label.indices), len(img))

        valid_loss = valid_loss.avg
        valid_acc = valid_acc.avg
        valid_f1 = valid_f1.avg
        print("Valid Loss %.4f | Valid Acc %.4f | f1 score %.4f" % (valid_loss, valid_acc, valid_f1))
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar("f1/valid", valid_f1, epoch)

        if valid_loss < best_valid_loss:
            print("New valid model for val loss! saving the model...")
            torch.save(model.state_dict(), save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_loss = valid_loss

        if valid_acc > best_valid_acc:
            print("New valid model for val accuracy! saving the model...")
            torch.save(model.state_dict(), save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_acc = valid_acc

        writer.flush()


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
    transformer = json_data["transformer"]

    PATH = args.save

    # setting path
    sys.path.append('/opt/ml/pstage01')
    from model import models, loss, metric
    from dataloader import mask
    from util import meter, transformers

    # setting device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # setting seed
    seed_all(seed=seed)

    # RGB mean, std
    mean, std = (0.56019358, 0.52410121, 0.501457), (0.23318603, 0.24300033, 0.24567522)

    transform = transformers.get_transforms(mean=mean, std=std, transform_type=transformer)

    dataset = mask.MaskBaseDataset(
        img_dir=img_dir
    )

    # train dataset과 validation dataset을 9:1 비율로 나눕니다.
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

    # 각 dataset에 augmentation 함수를 설정합니다.
    train_dataset.dataset.set_transform(transform['train'])
    val_dataset.dataset.set_transform(transform['val'])

    # training dataloader은 데이터를 섞어주어야 합니다. (shuffle=True)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

    valid_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )

    images, labels = next(iter(train_loader))

    model = get_model(model_name)
    model.cuda()

    criterion = loss.FocalLoss(gamma=loss_gamma)
    # criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay= weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    train(model, epochs, train_loader, valid_loader, PATH)
    model.cpu()
