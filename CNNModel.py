import sys
import cv2
import numpy as np
from math import pi
import keras
from keras.models import load_model
from keras.models import model_from_json
import json

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 72, 128, 3

class CNNModel():
    def __init__(self, json_model, weight):
        with open(json_model, 'r') as json_file:
            loaded_model_json = json.load(json_file)
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weight)

    def convert_img(self, image):
        return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

    def predict(self, img):
        img = self.convert_img(img)
        img = np.array(img)/255
        return self.model.predict(img.reshape(1,72,128,3))