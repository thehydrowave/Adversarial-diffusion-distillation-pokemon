# discriminator.py
import torch.nn as nn
import timm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=1)

    def forward(self, x):
        return self.backbone(x)
