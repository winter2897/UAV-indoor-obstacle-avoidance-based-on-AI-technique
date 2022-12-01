# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import time
import sys
import math

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

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

help = False

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print (textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime=time.process_time()
fps = 0

count = 0

def calculate_velocity(vx, vy):
    return math.sqrt(vx^2, vy^2)

def calculate_angle(vx_old, vy_old, vx_new, vy_new):
    

with open('data.csv', 'w') as f:
    while True:
        try:
            # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
            rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
            if (rawImage == None):
                print("Camera is not returning image, please check airsim for error messages")
                sys.exit(0)
            else:
                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
                count += 1
                filename = "imgs/" + str(count) + ".png"
                print(filename + "," + str(quad_vel.x_val) + "," + str(quad_vel.y_val))
                f.write(filename + "," + str(quad_vel.x_val) + "," + str(quad_vel.y_val) + '\n')
                cv2.imwrite(filename, png)
            
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
