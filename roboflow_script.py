#make sure required libraries are installed
'''
"%pip will install the package in the virtual environment where the current notebook kernel is running. While !pip will install the package in the base environment."
"Using ! allows to run commands like ls or pip or what you have available on your OS."
'''

# import the inference-sdk
from inference_sdk import InferenceHTTPClient

from zipfile import ZipFile

import os
from os import listdir

import cv2

#import numpy as np
#from pycocotools import mask as cocomask


# initialize the client
CLIENT = InferenceHTTPClient(

    #for each new version, the api url and key will be the same
    api_url="https://detect.roboflow.com",
    api_key="pOI9i83zbMXv4S4SJPzi"
)


print(os.path.exists('/home/earthsense/Downloads/temp_dataset.zip'))
with ZipFile('/home/earthsense/Downloads/temp_dataset.zip', 'r') as zipObj:
  os.makedirs("/home/earthsense/Documents/unzipped_temp_dataset", exist_ok=True)
  zipObj.extractall('/home/earthsense/Documents/unzipped_temp_dataset')
