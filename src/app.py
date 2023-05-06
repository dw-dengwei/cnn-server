from flask import Flask
from flask_cors import CORS
import json

app = Flask(__name__) 
CORS(app, supports_credentials=True)

@app.post('/get_model')
def hello():
    with open('res/model.json', 'r') as f:
        a = json.loads(f.read())
    return a


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000) 