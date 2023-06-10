from flask import Flask, Response, request
from flask_cors import CORS
import json
import numpy as np
from PIL import Image
from io import BytesIO
import torch.nn as nn
from collections import OrderedDict
import torch
import base64
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
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


class DIYModel(nn.Module):
    def __init__(self, dic):
        super(DIYModel, self).__init__()
        self.create_model(dic)

    def create_model(self, dic):
        model_layers = OrderedDict()
        for key, value in dic.items():
            if key.find("conv") != -1:
                model_layers[key]=nn.Conv2d(in_channels=value['in_channels'], out_channels=value['out_channels'], kernel_size=value['kernel_size'], padding=value['padding'])
            elif key.find("pool") !=-1:
                model_layers[key] = nn.MaxPool2d(value['stride'])
            elif key.find("relu") !=-1:
                model_layers[key] = nn.ReLU()
            elif key.find("flatten") != -1:
                model_layers[key] = nn.Flatten(1)
            elif key.find("output") != -1:
                model_layers[key] = nn.Linear(value['in_channels'], value['out_channels'])
        self.model = nn.Sequential(model_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def extract_layers(self):
        dic = {}
        dic['model'] = []
        for layer, name in zip(self.model, self.model._modules.keys()):
            if name.lower().find("conv") != -1 or name.lower().find("linear") != -1:
                dic['model'].append({"name": name, "kernel": layer.weight, "bias": layer.bias})
            else:
                dic['model'].append({"name": name})
        return dic

    def extract_feat(self, x):
        dic = {}
        dic["allOutputs"] = []
        assert x.shape[0] == 1
        for layer in self.model:
            x = layer(x)
            dic["allOutputs"].append(x[0])
        return dic


app = Flask(__name__)
CORS(app, supports_credentials=True)


def gen_tensor(shape):
    return np.random.rand(*shape).tolist()


@app.post('/get_model')
def get_model_base():
    model = BaseModel()
    # model.load_state_dict(torch.load("path"))
    model_layers = model.extract_layers()
    return json.dumps(model_layers)


@app.post('/get_feature_map')
def get_feature_map_base():
    # print(request.form['image'])
    base64_str = request.form['image'].split('base64,')[1]
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image).resize((64, 64)).convert('RGB')
    img_np = np.array(image)
    trans = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# 66.98
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
         ]
    )
    img_tensor = trans(img_np)

    model = BaseModel()
    # model.load_state_dict(torch.load("path"))
    # model(img_tensor)

    allOutputs = model.extract_feat(img_tensor.unsqueeze(0))

    inputImageArray = img_tensor.cpu().numpy().tolist()

    res = {
        'model': model.extract_layers(),
        'allOutputs': allOutputs,
        'inputImageArray': inputImageArray
    }
    return json.dumps(res)


@app.post('/train_diymodel')
def train_diymodel():
    img_bin = request.files.get('file').stream.read()
    img_np = np.array(Image.open(BytesIO(img_bin)))
    trans = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# 66.98
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
         ]
    )
    img_tensor = trans(img_np)

    train_data = CIFAR10("dataset", train=True, transform=trans,
                         download=True)
    # test_data = CIFAR10("dataset", train=False, transform=transform,
    #                     download=True)
    # train_size = int(0.8 * len(train_data))
    # val_size = len(train_data) - train_size
    # train_data, val_data = random_split(train_data, [train_size, val_size]
    #                                     , generator=torch.Generator().manual_seed(123))
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64)
    # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=64)
    # val_dataloader = DataLoader(val_data, batch_size=64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    model_params = OrderedDict()
    model_params["conv_1_1"] = {"in_channels": 3, "out_channels": 32, "kernel_size": 5, "padding": 2}

    model_params["pool_1_2"] = {"stride": 2}
    model_params["flatten_2_1"] = {},
    model_params["linear_3_1"]= {"in_channel": 1024, "out_channel": 10}

    model = DIYModel(model_params).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # best_test_accuracy = 0
    for current_epoch in range(100):
        model.train()
        for train_samples in train_dataloader:
            imgs, targets = train_samples
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if current_epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                allOutputs = model.extract_feat(img_tensor.unsqueeze(0))
                inputImageArray = img_tensor.cpu().numpy().tolist()
                model_layers = model.extract_layers()
                res = {
                    'allOutputs': allOutputs,
                    'inputImageArray': inputImageArray,
                    'model': model_layers
                }
                return json.dumps(res)  # 换成某个能传信息的函数


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
