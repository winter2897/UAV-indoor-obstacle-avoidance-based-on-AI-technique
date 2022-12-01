import sys
from controller.cnn.CNNController import CNNController

import airsim
import cv2
import time
import math
import argparse
from controller.Controller import Controller
from controller.rl.model import dqn
import numpy as np



client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
print("arming the drone...")
client.armDisarm(True)

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("takeoff failed - check Unreal message log for details")

cnn_controller = CNNController(client, 'controller/cnn/models/CNNModel.json', 'controller/cnn/models/CNNWeight.hdf5')
controller = Controller(client)


while True:
    try: 
        img = controller.getRGBImg()
        dqn_predict = dqn.model.predict(img.reshape(1,144,256,3))
        cnn_predict = cnn_controller.predict(img)
        print(dqn_predict)
        controller.take_action(np.argmax(dqn_predict))
        # time.sleep(0.5)
    except KeyboardInterrupt:
        break