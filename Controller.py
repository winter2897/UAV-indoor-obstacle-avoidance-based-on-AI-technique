import airsim
import math
from airsim import MultirotorClient
import sys
import cv2
import numpy as np

class Controller():
    def __init__(self, client):
        self.yaw = 0 
        self.vx = 0
        self.vy = 0
        self.velocity, self.angle = 1, 0
        self.client = client
    

    def take_off(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        print("arming the drone...")
        self.client.armDisarm(True)

        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed:
            print("taking off...")
            self.client.takeoffAsync().join()

    def getRGBImg(self):
        rawImage = self.client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            try:
                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(png, cv2.COLOR_RGBA2RGB)
                return img
            except:
                return np.zeros((144, 256, 3))

    def take_action(self, action):
        raise NotImplementedError()

    def moveByVolocity(self, velocity, angle):
        pitch, roll, self.yaw  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        self.yaw = (self.yaw + angle)
        self.vx = velocity *  math.cos(self.yaw)
        self.vy = velocity * math.sin(self.yaw)
        if (self.vx == 0 and self.vy == 0):
            self.vx = velocity * math.cos(self.yaw)
            self.vy = velocity * math.sin(self.yaw)
        self.client.moveByVelocityZAsync(self.vx, self.vy,-2, 0.5, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()