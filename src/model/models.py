import torch
from collections import OrderedDict
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ("conv_1_1", nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=0)),
            ("relu_1_2", nn.ReLU()),
            ("pool_1_3", nn.MaxPool2d(2)),

            ("conv_2_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_2_2", nn.ReLU()),
            ("pool_2_3", nn.MaxPool2d(2)),

            ("flatten_3_1", nn.Flatten(1)),
            ("output_4_1", nn.Linear(1960, 10)),
        ]))

    def forward(self, x):
        x = self.model(x)
        return x

    def extract_layers(self):
        ret_layer = {}

        ret_layer['layers'] = []
        for layer, name in zip(self.model, self.model._modules.keys()):
            if name.lower().find("conv") != -1 or name.lower().find("output") != -1:
                ret_layer['layers'].append({
                    "name": name, 
                    "kernel": layer.weight.detach().numpy().tolist(), 
                    "bias": layer.bias.detach().numpy().tolist()
                })
            else:
                ret_layer['layers'].append({"name": name})
        
        return ret_layer

    def extract_feat(self, x):
        ret = []
        assert x.shape[0] == 1
        for layer in self.model:
            x = layer(x)
            ret.append(x[0].detach().numpy().tolist())
        return ret


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ("conv_1_1", nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=0)),
            ("relu_1_2", nn.ReLU()),

            ("conv_2_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_2_2", nn.ReLU()),
            ("pool_2_3", nn.MaxPool2d(2)),

            ("conv_3_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_3_2", nn.ReLU()),

            ("conv_4_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_4_2", nn.ReLU()),
            ("pool_4_3", nn.MaxPool2d(2)),

            ("flatten_5_1", nn.Flatten(1)),
            ("output_6_1", nn.Linear(1690, 10)),
        ]))

    def forward(self, x):
        x = self.model(x)
        return x

    def extract_layers(self):
        ret_layer = {}

        ret_layer['layers'] = []
        for layer, name in zip(self.model, self.model._modules.keys()):
            if name.lower().find("conv") != -1 or name.lower().find("output") != -1:
                ret_layer['layers'].append({
                    "name": name, 
                    "kernel": layer.weight.detach().numpy().tolist(), 
                    "bias": layer.bias.detach().numpy().tolist()
                })
            else:
                ret_layer['layers'].append({"name": name})
        
        return ret_layer

    def extract_feat(self, x):
        ret = []
        assert x.shape[0] == 1
        for layer in self.model:
            x = layer(x)
            ret.append(x[0].detach().numpy().tolist())
        return ret

        
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ("conv_1_1", nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=0)),
            ("relu_1_2", nn.ReLU()),

            ("conv_2_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_2_2", nn.ReLU()),
            ("pool_2_3", nn.MaxPool2d(2)),

            ("conv_3_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_3_2", nn.ReLU()),

            ("conv_4_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_4_2", nn.ReLU()),
            ("pool_4_3", nn.MaxPool2d(2)),

            ("conv_5_1", nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0)),
            ("relu_5_2", nn.ReLU()),

            ("flatten_6_1", nn.Flatten(1)),
            ("output_7_1", nn.Linear(1210, 10)),
        ]))

    def forward(self, x):
        x = self.model(x)
        return x

    def extract_layers(self):
        ret_layer = {}

        ret_layer['layers'] = []
        for layer, name in zip(self.model, self.model._modules.keys()):
            if name.lower().find("conv") != -1 or name.lower().find("output") != -1:
                ret_layer['layers'].append({
                    "name": name, 
                    "kernel": layer.weight.detach().numpy().tolist(), 
                    "bias": layer.bias.detach().numpy().tolist()
                })
            else:
                ret_layer['layers'].append({"name": name})
        
        return ret_layer

    def extract_feat(self, x):
        ret = []
        assert x.shape[0] == 1
        for layer in self.model:
            x = layer(x)
            ret.append(x[0].detach().numpy().tolist())
        return ret