import os
import sys

import torch

from tqdm import tqdm

sys.path.append('/opt/ml/pstage01')
from model import loss, metric
from util import meter

def train(model, epochs, train_loader,criterion,optimizer,scheduler,device, save_path):
    """
    검증 셋 없이 학습
    """
    best_valid_acc = 0
    best_valid_loss = 100000

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = meter.AverageMeter(), meter.AverageMeter()
        for iter, (img, label, _) in enumerate(tqdm(train_loader)):
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
        print("Epoch %d | Train Loss %.4f | Train Acc %.4f" % (epoch, epoch_train_loss, epoch_train_acc))
        
        torch.save(model.state_dict(), save_path + f"{epoch:03}.ckpt")