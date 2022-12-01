"""
For connecting to the AirSim drone environment and testing API functionality
"""
import airsim
import math
from math import pi
import cv2

import os
import sys
import tempfile
import pprint

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")

cameraType = "scene"

for arg in sys.argv[1:]:
  cameraType = arg.lower()

cameraTypeMap = { 
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}

if (not cameraType in cameraTypeMap):
  printUsage()
  sys.exit(0)

print (cameraTypeMap[cameraType])

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

def interpret_angle(action, current_velocity, current_angle):
    velocity = current_velocity
    angle = current_angle
    if action.isdigit():
        velocity = 0.3 * int(action)
        angle = 0
    if (action == 'a'):
        angle = -4 * pi / 20
    elif (action == 's'):
        angle = -3 * pi / 20
    elif (action == 'd'):
        angle = -2 * pi / 20
    elif (action == 'f'):
        angle = - pi / 20
    elif (action == 'g'):
        angle = 0
    elif (action == 'h'):
        angle = pi / 20
    elif (action == 'j'):
        angle = 2 * pi / 20
    elif (action == 'k'):
        angle = 3 * pi / 20
    elif (action == 'l'):
        angle = 4 * pi / 10
    return velocity, angle


def control(velocity, angle):
    pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    print(yaw)
    yaw = (yaw + angle)
    vx = velocity *  math.cos(yaw);
    vy = velocity * math.sin(yaw);
    if (vx == 0 and vy == 0):
        vx = velocity * math.cos(yaw);
        vy = velocity * math.sin(yaw);
    client.moveByVelocityZAsync(vx, vy,-2, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()

action = ''
velocity, angle = 1, 0
count = 0
with open('data.csv', 'w') as f:
    while (True):
        try:
            action = airsim.wait_key('Action: ')
            velocity, angle = interpret_angle(action, velocity, angle)
            rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
            if (rawImage == None):
                print("Camera is not returning image, please check airsim for error messages")
                sys.exit(0)
            else:
                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
                print(client.getMultirotorState().rc_data.yaw)
                count += 1
                filename = "imgs/" + str(count) + ".png"
                print(filename + "," + str(quad_vel.x_val) + "," + str(quad_vel.y_val) + "," + str(velocity) + "," + str(angle))
                f.write(filename + "," + str(quad_vel.x_val) + "," + str(quad_vel.y_val) + "," + str(velocity) + "," + str(angle) + '\n')
                cv2.imwrite(filename, png)
            control(velocity, angle)
        except KeyboardInterrupt:
            break