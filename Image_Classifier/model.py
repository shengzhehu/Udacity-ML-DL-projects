from torchvision import models
from collections import OrderedDict
from torch import nn
import torch

def create_model(arch):
    """
    Creates NN model
    Parameters:
    - arch
    Return
    - NN model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = 'vgg11'
    model = models.__getattribute__(arch)(pretrained=True)


    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu1', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))


    model.classifier = classifier
    return model
    