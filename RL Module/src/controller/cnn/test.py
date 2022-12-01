from CNNController import CNNController

import airsim
import cv2
import time
import sys
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='models/CNNModel.json')
parser.add_argument('--weight', type=str, default='models/CNNWeight.hdf5')
args = parser.parse_args()

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

filename = 'test.png'

ccn_controller = CNNController(client, args.json, args.weight)

while True:
    try: 
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_RGBA2RGB)
            ccn_controller.control(img)
            # time.sleep(0.5)
    except KeyboardInterrupt:
        break
