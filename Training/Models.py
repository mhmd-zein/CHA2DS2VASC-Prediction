import torch
import timm

from utils import set_seed
from torchvision.models import efficientnet_b0, efficientnet_b4, efficientnet_v2_s, swin_v2_s, EfficientNet_B4_Weights, EfficientNet_B0_Weights, EfficientNet_V2_S_Weights, Swin_V2_S_Weights
from torch.nn import CrossEntropyLoss, Linear

class Custom_Net(torch.nn.Module):
    def __init__(self, num_classes=2, input_shape=[3,256,256], pretrained=False,
                 dropout=0.0, Adaptive_input=False, version='b0'):
        super(Custom_Net, self).__init__()
        self.version=version
        self.Adaptive_input=Adaptive_input

        if self.version=='b0':
            self.model  = efficientnet_b0(
            weights=(EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None),
                dropout=dropout
            )
            if self.Adaptive_input:
                old_conv = self.model.features[0][0]
                self.model.features[0][0] = torch.nn.Conv2d(
                    in_channels=4,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
            self.model.classifier[-1] = Linear(self.model.classifier[-1].in_features, num_classes)
        elif version=='b4':
            self.model  = efficientnet_b4(
                weights=(EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None),
                    dropout=dropout
                )
            self.model.classifier[-1] = Linear(self.model.classifier[-1].in_features, num_classes)
        elif version=='v2s':
            self.model  = efficientnet_v2_s(
                weights=(EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None),
                    dropout=dropout
                )
            self.model.classifier[-1] = Linear(self.model.classifier[-1].in_features, num_classes)

        
 
    def forward(self, x):               
        x = self.model(x)          
        return x

def build_model(num_classes=2, input_shape=[3,256,256], pretrained=True, dropout=0.0, adaptive_input=False,version='b0'):
    model = Custom_Net(
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        dropout=dropout,
        Adaptive_input=adaptive_input,
        version=version
    )
    return model