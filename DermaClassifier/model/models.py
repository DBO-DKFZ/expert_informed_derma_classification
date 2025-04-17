"""
Models https://paperswithcode.com/task/medical-image-classification
"""

from torch import nn
from torchvision import models


class ResNetModel(nn.Module):
    """Models a simple Convolutional Neural Network"""
    
    def __init__(self, model_layer="resnet50", num_classes=4, mode="majority"):
        """ initialize the network """
        super(ResNetModel, self).__init__()
        
        if model_layer == "resnet34":
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            in_feature = 512
        elif model_layer == "resnet50":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # DEFAULT # weights=ResNet50_Weights.IMAGENET1K_V1
            in_feature = 2048
        elif model_layer == "resnet101":
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            in_feature = 2048
        elif model_layer == "resnet152":
            self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
            in_feature = 2048

        self.resnet.fc = nn.Linear(in_feature, num_classes)
        self.soft_max = nn.Softmax(dim=1)  # nn.LogSoftmax(dim=1) if mode == "dist" else
    
    def forward(self, x):
        """ the forward propagation algorithm """
        x = self.resnet(x)
        x = self.soft_max(x)

        return x


class Efficient_Net(nn.Module):

    def __init__(self, model_block, num_classes=4, mode="ce") -> None:
        super(Efficient_Net, self).__init__()
        if model_block == "efficientnetB0":
            self.efficient_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_feature = 1280
        elif model_block == "efficientnetB1":
            self.efficient_net = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            in_feature = 1280
        elif model_block == "efficientnetB2":
            self.efficient_net = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            in_feature = 1408
        elif model_block == "efficientnetB4":
            self.efficient_net = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_feature = 1408
        elif model_block == "efficientnetB6":
            self.efficient_net = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
            in_feature = 1408
        elif model_block == "efficientnetB7":
            self.efficient_net = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            in_feature = 2560
        
        self.efficient_net.classifier[1] = nn.Linear(in_features=in_feature, out_features=num_classes,  bias=True)
        self.soft_max = nn.LogSoftmax(dim=1) if mode == "kl" else nn.Softmax(dim=1)
        
    def forward(self, x):
        """ the forward propagation algorithm """
        x = self.efficient_net(x)
        x = self.soft_max(x)

        return x


class ViT_net(nn.Module):

    def __init__(self, model_block, num_classes=4, mode="majority") -> None:
        super(ViT_net, self).__init__()
        if model_block == "vit16b":
            self.vit_net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_block == "vit32b":
            self.vit_net = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        
        self.vit_net.heads.head = nn.Linear(in_features=768, out_features=4, bias=True)
        self.soft_max = nn.Softmax(dim=1)  # nn.LogSoftmax(dim=1) if mode == "dist" else 
        
    def forward(self, x):
        """ the forward propagation algorithm """
        x = self.vit_net(x)
        x = self.soft_max(x)

        return x


def create_model(model: str, num_classes:int=4, mode:str="majority"):
    """ 
    Create model with corresponding structure. 
    
    :model: Name of modelstructure.
    :num_classes: Number of classes for which we want to solve the multiclass problem.
    :mode: majority or soft label 
    :return: torch model structure
    """

    if model in ["resnet34", "resnet50", "resnet101", "resnet152"]:
        return ResNetModel(model, num_classes, mode)
    elif model in ["efficientnetB0", "efficientnetB1", "efficientnetB2", "efficientnetB4", "efficientnetB6", "efficientnetB7"]:
        return Efficient_Net(model, num_classes, mode)
    elif model in ["vit16b", "vit32b"]:
        return ViT_net(model, num_classes, mode)
