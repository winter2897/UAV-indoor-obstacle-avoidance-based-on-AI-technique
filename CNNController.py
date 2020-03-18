import airsim
import cv2
import time
import sys
import math
from math import pi
import numpy as np

sys.path.append("controller")
sys.path.append("controller/cnn")
from CNNModel import CNNModel
# from ontroller import Controller
from Controller import Controller
# from controller.Controller import *
class CNNController(Controller):
    def take_action(self, action):
        velocity = 1
        if (action == 1):
            angle = -2 * pi / 20
        elif (action == 0):
            angle = 0
        elif (action == 2):
            angle = 2 * pi / 20
        self.moveByVolocity(velocity, angle)
        return velocity, angle
    
    def control(self, predict):
        self.take_action(np.argmax(predict))

cnn_model = CNNModel('controller/cnn/models/CNNModel.json', 'controller/cnn/models/CNNWeight.hdf5')

if __name__ == "__main__":
    cnn_controller = CNNController(airsim.MultirotorClient())
    cnn_controller.take_off()
    while True:
        try: 
            img = cnn_controller.getRGBImg()
            predict = cnn_model.predict(img)
            cnn_controller.control(predict)
        except KeyboardInterrupt:
            break