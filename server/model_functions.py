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
import imageio
import colorsys
from PIL import Image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_masks(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * 255,
                                  image[:, :, c])
    return image


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
        colors =  random_colors(5)
        print(colors)
        masked_image = image.copy()
        print(masked_image[500,500,:])
        red = [1.0,0.0,0.0]
        blue = [0.0,0.0,1.0]
        print(r['masks'].shape[2])
        #masked_image = apply_masks(masked_image, r['masks'].astype(int))
        alpha=0.5

        '''
        masks=r['masks'][:,:,0] 
        for i in range(1,r['masks'].shape[2]):
            masks = masks +r['masks'][:,:,i] 
        '''

        index_upp = [] 
        index_ruler = [] 
        for i in range(0, len(r['class_ids'])) : 
            if r['class_ids'][i] == 1 : 
                index_upp.append(i)
            if r['class_ids'][i] == 2 :
                index_ruler.append(i)
        masks_UPP = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
        masks_Ruler  =np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
        print( "index upp is ", index_upp , " and index ruler is" ,masks_Ruler )
        for i in range(len(index_upp)):
            masks_UPP = masks_UPP +r['masks'][:,:,index_upp[i]]
        for i in range(len(index_ruler)):
            masks_Ruler = masks_Ruler +r['masks'][:,:,index_ruler[i]]


        for c in range(3):
            masked_image[:, :, c] = np.where(masks_UPP.astype(int) == 1,
                                  masked_image[:, :, c] *
                                  (1 - alpha) + alpha * red[c]* 255,
                                  masked_image[:, :, c])
            masked_image[:, :, c] = np.where(masks_Ruler.astype(int) == 1,
                                  masked_image[:, :, c] *
                                  (1 - alpha) + alpha * blue[c]* 255,
                                  masked_image[:, :, c])
        print(masked_image)
        #plt.imshow(r['masks'].astype(int))
        print('server/static/images/tests/'+ str(test.camera_kurokesu) + '_segmented.png')

        pathn = str(test.camera_kurokesu)
        start = pathn.find('test') + 6
        end = pathn.find('.jpg', start)
        pathnn = pathn[start:end]
        print(pathnn)
        image_store_db = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented.png', masked_image) 
        test.Segmented_leftImage = 'tests/' + pathnn + '_segmented.png'

        
        test.NumberUlcers = str(list(r['class_ids']).count(1))
        #test.Ruler = "No"
        test.save()
        #print(r['class_ids'])
        cond.wait(0.1)
        logging.debug('Resource is available to consumer')


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
