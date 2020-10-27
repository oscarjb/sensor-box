import numpy as np
import tensorflow as tf

from tensorflow import keras
import os
import sys
import random
import math
import re
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from collections import OrderedDict
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from UPP_Contour_RULER_2020 import balloon

import logging
import threading
import time

from django.shortcuts import render
from django.http import HttpResponse


def consumer(cond, image,test):
    """wait for the condition and use the resource"""
    logging.debug('Starting consumer thread')
    with cond:
        #image = plt.imread('/home/evida/Documents/Sensorbox_V2/sensor-box/server/202.jpg')
        #model_maskRCNN.load_weights(os.path.join('server/static/model/mask_rcnn_type_0040.h5'), by_name=True)
        model_maskRCNN.load_weights(os.path.join(os.getcwd(),"../type20200227T1718/mask_rcnn_type_0040.h5"), by_name=True)

        results = model_maskRCNN.detect([image])
        r = results[0]
        if 2 in r['class_ids']: 
            test.Ruler = "Yes"
        else:
            test.Ruler = "No"

        
        test.NumberUlcers = str(list(r['class_ids']).count(1))
        #test.Ruler = "No"
        test.save()
        #print(r['class_ids'])
        cond.wait(0.1)
        logging.debug('Resource is available to consumer')
        context = {}
        context['hidden'] = True
        return render(request, "test_overview2.html", context)

def producer(cond):
    """set up the resource to be used by the consumer"""
    logging.debug('Starting producer thread')
    with cond:
        logging.debug('Making resource available')
        cond.notifyAll()


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s (%(threadName)-2s) %(message)s',
)

model = tf.keras.models.load_model(os.path.join('server/static/model/model.h5'), compile=False)
def pad(A, length=10000):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr


def predict(sample):
    sample_pad = np.array(pad(sample))
    return int(round(model.predict(np.array([sample_pad]))[0][0]))

config = balloon.FoodConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

DEVICE = "/cpu"
TEST_MODE = "inference"
config = InferenceConfig()
MODEL_DIR = os.path.join("logs_coco_resnet50_2020")
with tf.device(DEVICE):
    model_maskRCNN = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config)

print(model_maskRCNN)

def segment(image,test):
    condition = threading.Condition()
    c1 = threading.Thread(name='c1', target=consumer,args=(condition, image, test, ))
    c1.start()

    
    #model_maskRCNN.load_weights(os.path.join('server/static/model/mask_rcnn_type_0040.h5'), by_name=True)
    #results = model_maskRCNN.detect([image])
    return 1
