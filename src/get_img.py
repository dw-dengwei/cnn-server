import cv2
from torchvision.datasets import CIFAR10
import PIL
from PIL import Image
import numpy as np

test_id = []
train_data = CIFAR10("dataset", train=True, download=True)
for id, img in zip(train_data.targets, train_data.data):
    if len(test_id) == 10:
        break
    if id in test_id:
        continue
    test_id.append(id)
    cv2.imwrite(str(id)+".png", img)    
