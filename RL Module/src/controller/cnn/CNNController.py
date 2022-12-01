import sys
sys.path.append("../")

import cv2
import numpy as np
from math import pi
import keras
from keras.models import load_model
from keras.models import model_from_json
import json
from Controller import Controller

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 72, 128, 3

class CNNController(Controller):
    def __init__(self, client, json_model, weight, v_k_init = 1, s_k_init = 0):
        super().__init__(client)
        self.desired_forward_velocity = v_k_init
        self.desired_angular_velocity = s_k_init
        self.model = self.__load_model(json_model, weight)

    def __load_model(self, json_model, weight):
        with open(json_model, 'r') as json_file:
            loaded_model_json = json.load(json_file)
        model = model_from_json(loaded_model_json)
        model.load_weights(weight)
        return model

    def convert_img(self, image):
        return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

    def interpret_velocity_and_angle(self, action):
        velocity = 1
        if (action == 1):
            angle = -2 * pi / 20
        elif (action == 0):
            angle = 0
        elif (action == 2):
            angle = 2 * pi / 20
        print(velocity, angle)
        return velocity, angle

    def predict(self, img):
        img = self.convert_img(img)
        img = np.array(img)/255
        return self.model.predict(img.reshape(1,72,128,3))

    def control(self, img):
        outs = self.predict(img)
        v_k, s_k = self.interpret_velocity_and_angle(np.argmax(outs))
        self.moveByVolocity(v_k, s_k)