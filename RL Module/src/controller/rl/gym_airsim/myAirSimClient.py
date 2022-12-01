import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp
import sys

sys.path.append("../")
from Controller import Controller

from airsim import MultirotorClient
import airsim

class myAirSimClient(MultirotorClient):

    def __init__(self):        
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True)
        self.armDisarm(True)

        self.controller = Controller(self)
    
        self.home_pos = self.simGetGroundTruthKinematics().position
    
        self.home_ori = self.getOrientation()
        
        self.z = -2
    
    # def straight(self, duration, speed):
    #     pitch, roll, yaw  = airsim.to_eularian_angles(self.simGetVehiclePose().orientation)
    #     vx = math.cos(yaw) * speed
    #     vy = math.sin(yaw) * speed
    #     self.moveByVelocityZAsync(vx, vy, self.z, duration, airsim.DrivetrainType.ForwardOnly)
    #     start = time.time()
    #     return start, duration
    
    # def yaw_right(self, duration):
    #     self.rotateByYawRateAsync(30, duration)
    #     start = time.time()
    #     return start, duration
    
    # def yaw_left(self, duration):
    #     self.rotateByYawRateAsync(-30, duration)
    #     start = time.time()
    #     return start, duration
    
    
    def take_action(self, action):
		
        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        while self.simGetGroundTruthKinematics().position.z_val < -7.0:
            self.moveToZAsync(-6, 3)
            time.sleep(1)
            print(self.simGetGroundTruthKinematics().position.z_val, "and", x)
            x = x + 1
            if x > 10:
                return True        
        
    
        start = time.time()
        duration = 0 
        
        collided = False

        if action == 0:
            self.controller.moveByVolocity(1, 0)

            
            
        if action == 1:
            self.controller.moveByVolocity(1, math.pi / 10)
            
        if action == 2:
            self.controller.moveByVolocity(1, - math.pi / 10)

        if self.simGetCollisionInfo().has_collided:
            return True
            
        return collided
    
    def getRGBImg(self):
        filename = 'test.png'
        rawImage = self.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_RGBA2RGB)
            return img


    def AirSim_reset(self):
        
        self.reset()
        time.sleep(0.2)
        self.enableApiControl(True)
        self.armDisarm(True)
        time.sleep(1)
        self.moveToZAsync(self.z, 3) 
        time.sleep(3)
