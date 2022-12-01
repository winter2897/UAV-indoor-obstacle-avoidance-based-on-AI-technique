import json
import airsim
import cv2
import time
import sys
import math
from gym_airsim.myAirSimClient import myAirSimClient
import numpy as np
from RLController import dqn

sys.path.append('../cnn')
from CNNController import CNNController

dqn.load_weights('RLWeight_airsim-v1.h5f')

client = myAirSimClient()
print("arming the drone...")
client.armDisarm(True)

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("takeoff failed - check Unreal message log for details")

cnn_controller = CNNController(client, 'CNNModel.json', 'CNNWeight.hdf5')

count = 0
count_collision = 0
total_steps = 200
ensemble = False
cnn = True
rl = False
arr = [][]

def synthtic(count, count_collision, arr,type, total_steps):
    if count == 25:
        arr[0][type] = 25 - count_collision
    elif count == 50:
        arr[1][type] = 50 - count_collision
    elif count == 75:
        arr[2][type] = 75 - count_collision
    elif count == 100:
        arr[3][type] = 100 - count_collision
    return arr
# for i in range(100):
if ensemble:
    while count < total_steps:
        try: 
            img = client.getRGBImg()
            rl_predict = dqn.model.predict(img.reshape(1,144,256,3))
            cnn_predict = cnn_controller.predict(img)
            if cnn_predict[0, np.argmax(cnn_predict)] >= 0.98:
                print("CNN:", cnn_predict)
                v_k, s_k = cnn_controller.interpret_velocity_and_angle(np.argmax(cnn_predict))
                cnn_controller.moveByVolocity(v_k, s_k)
            else:
                print("RL", rl_predict)
                client.take_action(np.argmax(rl_predict))
            check_collision = client.getCollisionInfo().has_collided
            print("Collision infor: ", check_collision)
            if check_collision:
                count_collision += 1
            count += 1
            print("Count: ", count)
            print("Collection_cout: ", count_collision)
            arr = synthtic(count, count_collision, arr,0, total_steps)
        except KeyboardInterrupt:
            break
    print("Total collision: ", arr)
elif cnn:
# Test CNN
    while count < total_steps:
        try: 
            img = client.getRGBImg()
            cnn_predict = cnn_controller.predict(img)
            print("CNN:", cnn_predict)
            v_k, s_k = cnn_controller.interpret_velocity_and_angle(np.argmax(cnn_predict))
            cnn_controller.moveByVolocity(v_k, s_k)
            check_collision = client.getCollisionInfo().has_collided
            print("Collision infor: ", check_collision)
            if check_collision:
                count_collision += 1
            count += 1
            print("Count: ", count)
            print("Collection_cout: ", count_collision)
        except KeyboardInterrupt:
            break
    print("Total collision: ", count_collision)
elif rl:
# Test RL
    while count < total_steps:
        try: 
            img = client.getRGBImg()
            rl_predict = dqn.model.predict(img.reshape(1,144,256,3))
            print("RL", rl_predict)
            client.take_action(np.argmax(rl_predict))
            check_collision = client.getCollisionInfo().has_collided
            print("Collision infor: ", check_collision)
            if check_collision:
                count_collision += 1
            count += 1
            print("Count: ", count)
            print("Collection_cout: ", count_collision)
        except KeyboardInterrupt:
            break
    print("Total collision: ", count_collision)



