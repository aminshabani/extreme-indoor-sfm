import torch
import torch.nn as nn
import sys
from .unet import UNet
from .convmpn import get_convmpn_model
import logging
import torchvision as tv
logger = logging.getLogger('log')

class Classifier(nn.Module):
    def __init__(self, my_pretrained_model, input_dim, out_dim):
        super(Classifier, self).__init__()
        self.pretrained = my_pretrained_model
        # self.my_new_layers1 = nn.Sequential(nn.ReLU(), nn.Linear(input_dim*1, 4))
        self.my_new_layers2 = nn.Sequential(nn.Linear(16, 1))

    def forward(self, img1, img2, mask1, mask2, is_train=True):
        x1 = self.pretrained(torch.cat([img1, mask1], 1))
        # x1 = self.pretrained(img1)
        x2 = self.pretrained(torch.cat([img2, mask2], 1))
        # x2 = self.pretrained(img2)
        # if is_train:
            # x1 = torch.dropout(x1, 0.5, is_train)
            # x2 = torch.dropout(x2, 0.5, is_train)
        # x1 = self.my_new_layers1(x1)
        # x2 = self.my_new_layers1(x2)
        # x = self.my_new_layers2(torch.cat([x1, x2], 1))
        x = self.my_new_layers2(x1*x2)
        return x


def get_model(name, inp_dim=3, out_dim=1):
    if (name == 'unet'):
        model = UNet(inp_dim, out_dim, decoder=False)
    elif (name == 'unet-encoder'):
        model = UNet(inp_dim, None, decoder=False)
        model = Classifier(model, 16, out_dim)
    elif (name == 'resnet-encoder'):
        model = tv.models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(inp_dim,
                                64,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                bias=False)
        model = Classifier(model, 1000, out_dim)
    elif ('convmpn' in name):
        model = get_convmpn_model(name, inp_dim)
    else:
        logging.error("model type {} has not found".format(name))
        sys.exit(1)

    model = model.cuda()
    return model
