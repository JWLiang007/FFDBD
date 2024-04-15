import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
import sys
sys.path.append('./src')
import yaml
import abc
import torch.optim as optim



class Detector(nn.Module):

    def __init__(self,model='efb4'):
        super(Detector, self).__init__()
        if model != 'efb4':
            model_func = eval(f'models.{model}')
            self.net = model_func(pretrained=True)
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
            self.net.fc = nn.Linear(self.net.fc.in_features, 2)
            if hasattr(self.net,'AuxLogits'):
                self.net.AuxLogits.fc = nn.Linear(self.net.AuxLogits.fc.in_features, 2)
        else:
            self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        
        

    def forward(self,x):
        x=self.net(x)
        if hasattr(x,'logits') and hasattr(x,'aux_logits'):
            x = x.logits
            
        return x


