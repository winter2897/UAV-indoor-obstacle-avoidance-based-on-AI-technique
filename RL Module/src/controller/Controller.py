import airsim
import math

class Controller:
    def __init__(self, client):
        self.client = client
        self.yaw = 0 
        self.vx = 0
        self.vy = 0

    def moveByVolocity(self, velocity, angle):
        pitch, roll, self.yaw  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        self.yaw = (self.yaw + angle)
        self.vx = velocity *  math.cos(self.yaw)
        self.vy = velocity * math.sin(self.yaw)
        if (self.vx == 0 and self.vy == 0):
            self.vx = velocity * math.cos(self.yaw)
            self.vy = velocity * math.sin(self.yaw)
        self.client.moveByVelocityZAsync(self.vx, self.vy,-2, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()