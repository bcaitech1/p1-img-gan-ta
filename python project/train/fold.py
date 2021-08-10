import os
import sys

import torch

from tqdm import tqdm

sys.path.append('/opt/ml/pstage01')
from model import loss, metric
from util import meter

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def train(fold_count, model, epochs, train_loader, valid_loader,criterion,optimizer,scheduler,device,save_path):
    """
    k fold cross validation을 위한 함수
    """
    best_valid_acc = 0
    best_valid_loss = 100000
    save_path += f"{fold_count}/"

    createFolder(save_path)

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = meter.AverageMeter(), meter.AverageMeter()
        for iter, (img, label) in enumerate(tqdm(train_loader)):
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

            # Accuracy 계산
            pred_label = torch.max(pred_logit.data, 1)

            # pred_label  = pred_logit.argmax(1)
            acc = (pred_label.indices == label).sum().item() / len(label)

            epoch_train_loss.update(loss.item(), len(img))
            epoch_train_acc.update(acc, len(img))
        scheduler.step()

        #             if (iter %10 == 0) or (iter == len(train_loader)-1):
        #                 print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f" %(iter, len(train_loader), train_loss, train_acc))

        epoch_train_loss = epoch_train_loss.avg
        epoch_train_acc = epoch_train_acc.avg
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



        if valid_loss < best_valid_loss:
            print("New valid model for val loss! saving the model...")
            torch.save(model.state_dict(), save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_loss = valid_loss

        if valid_acc > best_valid_acc:
            print("New valid model for val accuracy! saving the model...")
            torch.save(model.state_dict(), save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_acc = valid_acc

        # writer.flush()
