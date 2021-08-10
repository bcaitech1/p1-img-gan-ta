import os
import sys

import torch

sys.path.append('/opt/ml/pstage01')
from model import loss, metric
from util import meter


def train(model, epochs, train_loader, valid_loader, save_path,criterion,optimizer,scheduler,device, wandb, mixup = False, cutmix = False):
    best_valid_acc = 0
    best_valid_loss = 100000

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = meter.AverageMeter(), meter.AverageMeter()
        for iter, (img,label, path) in enumerate(train_loader):
            optimizer.zero_grad()

            # label1 증강시키기(mixup)
            if mixup == True:
                re_img, label1, label2, alpha, beta = train_loader.dataset.dataset.get_mixup_img(path,label,1,2)
                re_img, label1, label2 = re_img.type(torch.FloatTensor).to(device), label1.to(device), label2.to(device)

                if len(re_img) != 0:
                    optimizer.zero_grad()
                    pred_logit = model(re_img)
                    loss = criterion(pred_logit, label1) * alpha + criterion(pred_logit, label2) * beta
                    loss.backward()
                    optimizer.step()
            
            #label1 증강시키기(cutmix)
            if cutmix == True:
                re_img1, re_img2 ,label1, label2, alpha, beta = train_loader.dataset.dataset.get_cutmix_img(path,label,1,2)
                re_img1, re_img2, label1, label2 = re_img1.type(torch.FloatTensor).to(device),re_img2.type(torch.FloatTensor).to(device), label1.to(device), label2.to(device)

                if len(re_img1) != 0:
                    optimizer.zero_grad()
                    pred_logit = model(re_img1)
                    loss = criterion(pred_logit, label1) * alpha + criterion(pred_logit, label2) * beta
                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    pred_logit = model(re_img2)
                    loss = criterion(pred_logit, label1) * beta + criterion(pred_logit, label2) * alpha
                    loss.backward()
                    optimizer.step()

            img, label = img.type(torch.FloatTensor).to(device), label.to(device)

            # 모델에 이미지 forward
            model.train()
            pred_logit = model(img)

            # loss 값 계산
            loss = criterion(pred_logit,label)

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accuracy 계산
            pred_label = torch.max(pred_logit.data,1)

            # pred_label  = pred_logit.argmax(1)
            acc = (pred_label.indices == label).sum().item() / len(label)

            epoch_train_loss.update(loss.item(), len(img))
            epoch_train_acc.update(acc, len(img))

            train_loss = loss.item()
            train_acc = acc
            
            if (iter %10 == 0) or (iter == len(train_loader)-1):
                print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f" %(iter, len(train_loader), train_loss, train_acc))

        epoch_train_loss = epoch_train_loss.avg
        epoch_train_acc = epoch_train_acc.avg
        print("Epoch %d | Train Loss %.4f | Train Acc %.4f"%(epoch,epoch_train_loss, epoch_train_acc))
        
        valid_loss, valid_acc, valid_f1 = meter.AverageMeter(), meter.AverageMeter(), meter.AverageMeter()
        model.eval()
        
        for img, label, path in valid_loader:
            img, label = img.type(torch.FloatTensor).to(device), label.to(device)

            with torch.no_grad():
                pred_logit = model(img)
            loss = criterion(pred_logit, label) 
            pred_label = torch.max(pred_logit.data,1)

            acc = (pred_label.indices == label).sum().item() / len(label)

            valid_loss.update(loss.item(), len(img))
            valid_acc.update(acc, len(img))
            valid_f1.update(metric.f1_loss(label, pred_label.indices), len(img))


        valid_loss = valid_loss.avg
        valid_acc = valid_acc.avg
        valid_f1 = valid_f1.avg
        print("Valid Loss %.4f | Valid Acc %.4f | f1 score %.4f" %(valid_loss, valid_acc, valid_f1))
        
        if valid_loss < best_valid_loss:
            print("New valid model for val loss! saving the model...")
            torch.save(model.state_dict(),save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_loss = valid_loss
        
        if valid_acc > best_valid_acc:
            print("New valid model for val accuracy! saving the model...")
            torch.save(model.state_dict(),save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_acc = valid_acc
