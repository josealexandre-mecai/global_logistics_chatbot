import sys
sys.path.append('src')  # Add 'src' folder to the system path

from bert_trainer import BertTrainer

import argparse
from flask import Flask, jsonify, request, render_template
import os
import re
import joblib
import socket
import json
import numpy as np
import pandas as pd

app = Flask(__name__)
model = BertTrainer(intents_json='global_logistics.json')

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    #user_message = request.form['user_input']
    user_message = request.json['user_input']
    print(user_message)
    intent, confidence = model.predict(model_version='1.0', model_name='logistics', data=user_message)
    print(intent, confidence)
    response = model.get_response(intent)
    print(response)
    #return json.dumps({'bot_response': response})
    return jsonify({"bot_response": response})

if __name__ == '__main__':
    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)
