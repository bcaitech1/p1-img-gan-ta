import os
import sys

import torch

sys.path.append('/opt/ml/pstage01')
from model import loss, metric
from util import meter

def train(model, epochs, train_loader, valid_loader,criterion,optimizer,scheduler,device, save_path, wandb):
    """
    multi head classification 학습
    """
    best_valid_acc = 0
    best_valid_loss = 100000

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = meter.AverageMeter(), meter.AverageMeter()
        for iter, (img, mask_label,gender_label,age_label, path) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            img = img.type(torch.FloatTensor).to(device)

            mask_label = mask_label.to(device)
            gender_label = gender_label.to(device)
            age_label = age_label.to(device)

            pred_logit = model(img) # mask, gender, age

            mask_loss = criterion(pred_logit[0], mask_label)
            gender_loss = criterion(pred_logit[1], gender_label)
            age_loss = criterion(pred_logit[2], age_label)

            loss =  (mask_loss + gender_loss + age_loss) / 3

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accuracy 계산
            mask_pred_label = torch.max(pred_logit[0].data, 1)
            gender_pred_label = torch.max(pred_logit[1].data, 1)
            age_pred_label = torch.max(pred_logit[2].data, 1)

            # pred_label  = pred_logit.argmax(1)
            mask_acc = (mask_pred_label.indices == mask_label).sum().item() / len(mask_label)
            gender_acc = (gender_pred_label.indices == gender_label).sum().item() / len(gender_label)
            age_acc = (age_pred_label.indices == age_label).sum().item() / len(age_label)

            acc = (mask_acc + gender_acc + age_acc) / 3
            loss = (mask_loss + gender_loss + age_loss) / 3

            epoch_train_loss.update(loss.item(), len(img))
            epoch_train_acc.update(acc, len(img))
            
            if (iter %100 == 0) or (iter == len(train_loader)-1):
                print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f" %(iter, len(train_loader), epoch_train_loss.avg, epoch_train_acc.avg))

        epoch_train_loss = epoch_train_loss.avg
        epoch_train_acc = epoch_train_acc.avg
        wandb.log({"Loss/train": epoch_train_loss, "Acc/train": epoch_train_acc})
        print("Epoch %d | Train Loss %.4f | Train Acc %.4f" % (epoch, epoch_train_loss, epoch_train_acc))

        valid_loss, valid_acc, valid_f1 = meter.AverageMeter(), meter.AverageMeter(), meter.AverageMeter()
        model.eval()

        for img, mask_label,gender_label,age_label,_ in valid_loader:
            img = img.type(torch.FloatTensor).to(device)

            mask_label = mask_label.to(device)
            gender_label = gender_label.to(device)
            age_label = age_label.to(device)

            with torch.no_grad():
                pred_logit = model(img)

            mask_loss = criterion(pred_logit[0], mask_label)
            gender_loss = criterion(pred_logit[1], gender_label)
            age_loss = criterion(pred_logit[2], age_label)

            mask_pred_label = torch.max(pred_logit[0].data, 1)
            gender_pred_label = torch.max(pred_logit[1].data, 1)
            age_pred_label = torch.max(pred_logit[2].data, 1)

            mask_acc = (mask_pred_label.indices == mask_label).sum().item() / len(mask_label)
            gender_acc = (gender_pred_label.indices == gender_label).sum().item() / len(gender_label)
            age_acc = (age_pred_label.indices == age_label).sum().item() / len(age_label)

            mask_f1 = metric.f1_loss(mask_label, mask_pred_label.indices)
            gender_f1 =  metric.f1_loss(mask_label, mask_pred_label.indices)
            age_f1 =  metric.f1_loss(mask_label, mask_pred_label.indices)
            f1 = (mask_f1 + gender_f1 + age_f1) / 3

            acc = (mask_acc + gender_acc + age_acc) / 3
            loss = (mask_loss + gender_loss + age_loss) / 3

            valid_loss.update(loss.item(), len(img))
            valid_acc.update(acc, len(img))
            valid_f1.update(f1, len(img))

        valid_loss = valid_loss.avg
        valid_acc = valid_acc.avg
        valid_f1 = valid_f1.avg
        wandb.log({"Loss/valid": valid_loss, "Acc/valid": valid_acc, "f1/valid": valid_f1})
        print("Valid Loss %.4f | Valid Acc %.4f | f1 score %.4f" % (valid_loss, valid_acc, valid_f1))

        if valid_loss < best_valid_loss:
            print("New valid model for val loss! saving the model...")
            torch.save(model.state_dict(), save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_loss = valid_loss

        if valid_acc > best_valid_acc:
            print("New valid model for val accuracy! saving the model...")
            torch.save(model.state_dict(), save_path + f"{epoch:03}_loss_{valid_loss:4.2}_acc_{valid_acc:4.2}.ckpt")
            best_valid_acc = valid_acc
