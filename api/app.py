
from flask import Flask

app = Flask(__name__)

@app.route('/predict/<domain>')
def hello(domain):
    return predict(domain)
