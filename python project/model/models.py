import torch
import torch.nn as nn

from torchvision import models

import timm


class MyResNext(nn.Module):
    """
    ResNext pretrained
    """
    def __init__(self, num_classes: int = 18):
        super(MyResNext, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=True)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x

class MyEfficentNetb4(nn.Module):
    """
    EfficentNetb4 pretrained
    """
    def __init__(self, num_calsses : int = 18):
        super(MyEfficentNetb4,self).__init__()
        self.backbone = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.backbone.classifier = nn.Linear(in_features=1792, out_features=num_calsses, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x
    
class MyVgg16(nn.Module):
    """
    Vgg16 pretrained
    """
    def __init__(self, num_calsses : int = 18):
        super(MyVgg16,self).__init__()
        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier[6] = nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x

class MyMultiResNext(nn.Module):
    """
    model for multi head classification
    """
    def __init__(self,mask_num_calsses : int = 3, gender_num_calsses : int = 2, age_num_calsses : int = 3):
        super(MyMultiResNext, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=True)
        self.mask_classifier = nn.Linear(in_features=1000, out_features=mask_num_calsses, bias=True)
        self.gender_classifier = nn.Linear(in_features=1000, out_features=gender_num_calsses, bias=True)
        self.age_classifier = nn.Linear(in_features=1000, out_features=age_num_calsses, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        z = self.age_classifier(x)
        y = self.gender_classifier(x)
        x = self.mask_classifier(x)

        return x, y, z