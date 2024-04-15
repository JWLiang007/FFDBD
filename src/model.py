import torch
from torch import nn
import torchvision
import torchvision.models as models
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM


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
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=SAM(self.parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)
        
        

    def forward(self,x):
        x=self.net(x)
        if hasattr(x,'logits') and hasattr(x,'aux_logits'):
            x = torch.cat((x.logits, x.aux_logits),0)
            
        return x
    
    def training_step(self,x,target,sign=1):
        _target = target.detach().clone()
        for i in range(2):
            pred_cls=self(x)
            if len(pred_cls) != len(target):
                target = torch.cat((target,target),0)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=sign*loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        if len(pred_cls) != len(_target):
            return pred_first[:len(_target)]
        return pred_first
    
    def training_step_abl(self,x,target,ascent_type,**params):
        _target = target.detach().clone()
        for i in range(2):
            pred_cls=self(x)
            if len(pred_cls) != len(target):
                target = torch.cat((target,target),0)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            if ascent_type == 'LGA':
                loss_ascent = torch.sign(loss_cls - params['gamma']) * loss_cls
            elif ascent_type == 'Flooding':
                loss_ascent = (loss_cls - params['flooding']).abs() + params['flooding']
            self.optimizer.zero_grad()
            loss_ascent.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        if len(pred_cls) != len(_target):
            return pred_first[:len(_target)],loss_ascent.detach().clone()
        return pred_first,loss_ascent.detach().clone()
    
