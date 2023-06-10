from flask import Flask, Response, request
from flask_cors import CORS
import json
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms


app = Flask(__name__) 
CORS(app, supports_credentials=True)


def gen_tensor(shape, sub=3):
    return (np.random.rand(*shape) - sub).tolist()


@app.post('/get_model')
def get_model():
    layers = [
        {
            'name': 'conv_1_1',
            'kernel': gen_tensor((10, 3, 3, 3), 2),
            'bias': gen_tensor((10,))
        },
        {
            'name': 'relu_1_2',
        },

        {
            'name': 'conv_2_1',
            'kernel': gen_tensor((10, 10, 3, 3)),
            'bias': gen_tensor((10,))
        },
        {
            'name': 'relu_2_2',
        },
        {
            'name': 'pool_2_3',
        },

        {
            'name': 'conv_3_1',
            'kernel': gen_tensor((10, 10, 3, 3)),
            'bias': gen_tensor((10,))
        },
        {
            'name': 'relu_3_2',
        },

        {
            'name': 'conv_4_1',
            'kernel': gen_tensor((10, 10, 3, 3)),
            'bias': gen_tensor((10,))
        },
        {
            'name': 'relu_4_2',
        },
        {
            'name': 'pool_4_3',
        },

        {
            'name': 'flatten_5_1',
        },

        {
            'name': 'output_6_1',
            'kernel': gen_tensor((10, 1690)),
            'bias': gen_tensor((10,))
        },
    ]
    jsonModel = {'layers': layers}
    return json.dumps(jsonModel)
    

@app.post('/get_feature_map')
def get_feature_map():
    # img_bin = request.files.get('file').stream.read()
    # img_np = np.array(Image.open(BytesIO(img_bin)))
    # img_tensor = transforms.ToTensor()(img_np)
   
    # model(img_tensor) 
    
    allOutputs = [
        (10, 62, 62),
        (10, 62, 62),
        (10, 60, 60),
        (10, 60, 60),
        (10, 30, 30),
        (10, 28, 28),
        (10, 28, 28),
        (10, 26, 26),
        (10, 26, 26),
        (10, 13, 13),
        (1690,),
        (10,)
    ]

    # inputImageArray = img_tensor.cpu().numpy().tolist()
    allOutputs = [gen_tensor(shape) for shape in allOutputs]

    res = {
        'allOutputs': allOutputs,
        # 'inputImageArray': inputImageArray
    }

    return json.dumps(res)


@app.post('/get_input_image_array')
def get_input_image_array():
    img_tensor = gen_tensor((3, 64, 64))

    return json.dumps({
        'inputImageArray': img_tensor
    })

    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001) 
