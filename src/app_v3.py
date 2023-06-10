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


app = Flask(__name__)
CORS(app, supports_credentials=True)
trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([64, 64]),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# 66.98
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)
train_data = CIFAR10("dataset", train=True, transform=trans, download=True)
train_dataloader = None
device = torch.device("cpu")
loss_fn = nn.CrossEntropyLoss()
train_model = None


@app.post('/init')
def init():
    global train_dataloader, train_model
    global optimizer
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64)
    train_model = BaseModel()
    optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-1)
    return 'ok'
    

@app.post('/get_feature_map')
def train_basemodel():
    base64_str = request.form['image'].split('base64,')[1]
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image).resize((64, 64)).convert('RGB')
    img_np = np.array(image)
    img_tensor = trans(img_np)

    # boptimizer = torch.optim.SGD(train_model.parameters(), lr=1e-3)
    # best_test_accuracy = 0
    epochs = 1
    iter_num = 10
    
    for current_epoch in range(epochs):
        train_model.train()
        cur_iter = 0
        for train_samples in train_dataloader:
            imgs, targets = train_samples
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = train_model(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_iter += 1
            if cur_iter > iter_num:
                break
            
    with torch.no_grad():
        train_model.eval()
        allOutputs = train_model.extract_feat(img_tensor.unsqueeze(0))
        inputImageArray = img_tensor.cpu().numpy().tolist()
        model_layers = train_model.extract_layers()
        res = {
            'allOutputs': allOutputs,
            'inputImageArray': inputImageArray,
            'model': model_layers
        }
        
    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
