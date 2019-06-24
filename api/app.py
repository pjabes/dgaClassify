
from flask import Flask
import sys
sys.path.append('../models')
from model import hello_world as hello_world 

app = Flask(__name__)

@app.route('/predict/<domain>')
def hello(domain):
    return predict(domain)

if __name__ == '__main__':
    app.run()

