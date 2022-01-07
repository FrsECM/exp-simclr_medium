import torch.nn as nn
import torch
from torch.nn.modules import activation
from torch.nn.modules.activation import LeakyReLU
from torchvision.models import resnet18
import numpy as np
from torchvision.models.resnet import resnet34, resnet50

global BACKBONES
BACKBONES={'resnet18':resnet18(),'resnet34':resnet34(),'resnet50':resnet50()}

class SimCLRModel(nn.Module):
    Name='SimCLRModel'

    def __init__(self,x_depth=1,x_H=64,x_W=64,z_dim:int=2,backbone='resnet18'):
        super(SimCLRModel,self).__init__()
        assert backbone in BACKBONES.keys(),f"backbone {backbone} should be in {list(BACKBONES.keys())}"
        model=BACKBONES[backbone]
        model.conv1 = nn.Conv2d(x_depth, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model.fc=nn.Identity()
        ######################## Encoder
        self.z_dim=z_dim
        self.x_depth=x_depth
        self.inputShape=[x_depth,x_H,x_W]
        self.encoder=model
        
        # We get the h_size.
        self.h_dim=self._get_conv_outsize(self.inputShape)
        # We create the Projetion Head
        self.proj=nn.Sequential(
            nn.Linear(self.h_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.z_dim)
        )

    def _get_conv_outsize(self,shape):
        with torch.no_grad():
            b=2
            self.conv_out_shape = self.encoder(torch.zeros(b, *shape)).size()
        return int(np.prod(self.conv_out_shape)//b)
    
    def encode(self,x):
        BatchSize=x.size()[0]
        for layer in self.encoder.children():
                x=layer(x)
        h=x.view(BatchSize,-1)
        return h
    
    def forward(self, x):
        h=self.encode(x)
        z=self.proj(h)
        return h,z

if __name__=='__main__':
    model=SimCLRModel(3,32,32,128)
    s=model(torch.ones((10,3,32,32)))