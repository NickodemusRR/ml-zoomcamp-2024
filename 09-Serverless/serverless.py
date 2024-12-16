#!/usr/bin/env python
# coding: utf-8

import os
import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()

    stream = BytesIO(buffer)
    img = Image.open(stream)

    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)

    return img

def input_preprocess(x):

    return x / 255.

MODEL_NAME = os.getenv('MODEL_NAME', 'model_2024_hairstyle.tflite')

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(200,200))

    x = np.array(img, dtype='float32') 
    X = np.array([x])
    X = input_preprocess(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_index)

    return float(prediction[0, 0])


def lambda_handler(event, context):
    url = event['url']
    prediction = predict(url)

    result = {
        'prediction': prediction
    }

    return result






