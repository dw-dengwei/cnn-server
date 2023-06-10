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
import copy
from threading import Thread
import threading, ctypes
import asyncio
from model.models import Model1, Model2, Model3


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
model_list = []


def train(model_id):
    global train_dataloader, train_model
    global optimizer, model_list
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64)
    if model_id == 1:
        train_model = Model1()
    elif model_id == 2:
        train_model = Model2()
    elif model_id == 3:
        train_model = Model3()
    else: # default
        train_model = Model1()
        
    optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-1)

    model_list.clear()
    epochs = 100
    iter_num = 10
    cur_iter = 0
    
    print('init complelte, training...')
    
    torch.autograd.set_detect_anomaly(True)
    
    for current_epoch in range(epochs):
        train_model.train()
        
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
            if cur_iter % iter_num == 0:
                while len(model_list) ==  50:
                    continue
                model_list.append(copy.deepcopy(train_model))                
                print('push', f'cur={len(model_list)}')


class KThread(Thread):
    def __init__(self, *params, **known):
        super(KThread, self).__init__(*params, **known)
        parent_thread = threading.current_thread()
        self.is_killed = False
        self.child_threads = []
        if hasattr(parent_thread, 'child_threads'):
            parent_thread.child_threads.append(self)

    def _raise_exc(self, exc_obj):
        if not self.is_alive():
            return

        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.ident), ctypes.py_object(exc_obj))
        if res == 0:
            raise RuntimeError("Not existent thread id.")
        elif res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(self.ident, None)
            raise SystemError("PyThreadState_SetAsyncExc failed.")

    def kill(self):
        if hasattr(self, 'child_threads'):
            for child_thread in self.child_threads:
                if child_thread.is_alive():                    
                    print('killed')
                    child_thread.kill()
        self._raise_exc(SystemExit)
        self.is_killed = True
                 

t1 = None       


@app.post('/init/<model_id>/')
def init(model_id):
    model_id = int(model_id)
    print(f'model id = {model_id}')
    global t1
    if t1:
        t1.kill()
        t1 = KThread(target=train, args=(model_id,))
    else:
        t1 = KThread(target=train, args=(model_id, ))
        
    model_list.clear()    
    
    t1.start()
            
    return 'ok'
    

@app.post('/get_feature_map')
def train_basemodel():
    global model_list
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([64, 64]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# 66.98
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )
    base64_str = request.form['image'].split('base64,')[1]
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image).resize((64, 64)).convert('RGB')
    img_np = np.array(image)
    img_tensor = trans(img_np)

    while len(model_list) == 0:
        continue

    print('pop', f'cur={len(model_list)}')
    model = model_list.pop(0)
            
    with torch.no_grad():
        model.eval()
        allOutputs = model.extract_feat(img_tensor.unsqueeze(0))
        inputImageArray = img_tensor.cpu().numpy().tolist()
        model_layers = model.extract_layers()
        res = {
            'allOutputs': allOutputs,
            'inputImageArray': inputImageArray,
            'model': model_layers
        }

    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
