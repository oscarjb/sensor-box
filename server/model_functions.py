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
from UPP_Contour_RULER_2020 import granulation
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
        print(" Detection of UPP and Ulcer:  Done")
        ##################################  Ruler Detection ################################################
        if 2 in r['class_ids']: 
            test.Ruler = "Yes"
        else:
            test.Ruler = "No"

        ################################## Detection of Number of Ulcers ####################################

        test.NumberUlcers = str(list(r['class_ids']).count(1))
        
        ################################## UPP and Ruler segmentation #######################################
        
        masked_image = image.copy()
        red = [1.0,0.0,0.0] # Red for UPP 
        blue = [0.0,0.0,1.0] # Blue for Ruler
        alpha=0.5
        index_upp = [] 
        index_ruler = [] 
        for i in range(0, len(r['class_ids'])) : 
            if r['class_ids'][i] == 1 : 
                index_upp.append(i)
            if r['class_ids'][i] == 2 :
                index_ruler.append(i)
        masks_UPP = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
        masks_Ruler  =np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
        
        for i in range(len(index_upp)):
            masks_UPP = masks_UPP +r['masks'][:,:,index_upp[i]]
        for i in range(len(index_ruler)):
            masks_Ruler = masks_Ruler +r['masks'][:,:,index_ruler[i]]

        masks_UPP[masks_UPP >=1 ] = 1
        masks_Ruler[masks_Ruler >=1 ] = 1

        for c in range(3):
            masked_image[:, :, c] = np.where(masks_UPP.astype(int) == 1,
                                  masked_image[:, :, c] *
                                  (1 - alpha) + alpha * red[c]* 255,
                                  masked_image[:, :, c])
            masked_image[:, :, c] = np.where(masks_Ruler.astype(int) == 1,
                                  masked_image[:, :, c] *
                                  (1 - alpha) + alpha * blue[c]* 255,
                                  masked_image[:, :, c])
        print(" Segmentation of UPP and Ulcer:  Done")
        #Saving the new segmented image
        pathn = str(test.camera_kurokesu)
        start = pathn.find('test') + 6
        end = pathn.find('.jpg', start)
        pathnn = pathn[start:end]
        image_store_db = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented.png', masked_image) 
        test.Segmented_leftImage = 'tests/' + pathnn + '_segmented.png'
        print(" Storing segmented image of UPP and Ruler:  Done")

        ################################## Granulation classification & percentage ################################################
        masked_g = image.copy()
        model_granulation.load_weights(os.path.join(os.getcwd(),"../Granulation_model/mask_rcnn_type_0019.h5"), by_name=True)
        results_g = model_granulation.detect([image])
        r_g = results_g[0]

        print(" Detection of granulation:  Done")
        print(r_g)
        masks_granulation  =np.zeros((r_g['masks'].shape[0], r_g['masks'].shape[1]))
        for i in range(len(r_g['class_ids'])):
            masks_granulation = masks_granulation +r_g['masks'][:,:,i]

        masks_granulation[masks_granulation >=1 ] = 1
        for c in range(3):
            masked_g[:, :, c] = np.where(masks_granulation.astype(int) == 1,
                                  masked_g[:, :, c] *
                                  (1 - alpha) + alpha * red[c]* 255,
                                  masked_g[:, :, c])

        print(" Segmentation of granulation:  Done")

        pathn = str(test.camera_kurokesu)
        start = pathn.find('test') + 6
        end = pathn.find('.jpg', start)
        pathnn = pathn[start:end]
        image_store_db_g = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_g.png', masked_g) 
        test.Segmented_leftImage_g = 'tests/' + pathnn + '_segmented_g.png'
        print(" Storing segmented granulation:  Done")
        
        test.TissueTypes = ''
        test.TissueTypes =  test.TissueTypes + 'Granulation'
        

        print(" Measuring percentage of Granulation:  Start")
        number_g= 0
        number_upp=sum(sum(masks_UPP.astype(int)))
        
        print(number_upp)
        print(sum(sum(masks_granulation.astype(int))))
        union = masks_UPP.astype(int) + masks_granulation.astype(int)
        zeross = np.zeros((union.shape[0],union.shape[1]))
        intersection = sum(sum(np.where(union==2 , union, zeross)))
        print(intersection)
        '''
        for i in range(masks_UPP.shape[0]):
            for j in range(masks_UPP.shape[1]):
                print(i,j)
                if (masks_UPP.astype(int)[i,j]==1 and masks_granulation.astype(int)[i,j]==1 ):
                    number_g +=1
        if number_upp!=0:            
            Percentage_g = number_g *100 / number_upp
            '''
        if number_upp!=0:            
            Percentage_g = intersection *100 / (2*number_upp)
        else: 
            Percentage_g = 0
        
        print(Percentage_g)


        
        
        test.Granulation =  str("{:.2f}".format(Percentage_g)) + ' %'
        '''
        masked_g2 = image.copy()
        for c in range(3):
            masked_g2[:, :, c] = np.where(np.where(union==2, union, zeross).astype(int) == 2,
                                  masked_g2[:, :, c] *
                                  (1 - alpha) + alpha * red[c]* 255,
                                  masked_g2[:, :, c])
        #test_g = imageio.imsave('test_g.png', masked_g2) 
        np.savetxt("foo.csv", masks_granulation.astype(int), delimiter=",") 
        print( masks_granulation[2500:3000,1200:2000])
        test_g = imageio.imsave('test_g.png', masks_granulation.astype(int))  
        '''        
        test_g = imageio.imsave('test_g.png', np.where(union==2 , union, zeross).astype(int)) 

        test_upp = imageio.imsave('test_upp.png', masks_UPP.astype(int))             
        print(" Measuring percentage of Granulation:  Done")
        ################################## Saving the patient data ################################################
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
config_g = granulation.FoodConfig()
config_g.display()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

DEVICE = "/cpu"
TEST_MODE = "inference"
config = InferenceConfig()
#config_g = InferenceConfig()
config_g.GPU_COUNT = 1 
config_g.IMAGES_PER_GPU = 1 
config_g.BATCH_SIZE = 1 

MODEL_DIR = os.path.join("logs_coco_resnet50_2020")
with tf.device(DEVICE):
    model_maskRCNN = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config)
with tf.device(DEVICE):
    model_granulation = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config_g)


def segment(image,test):
    condition = threading.Condition()
    c1 = threading.Thread(name='c1', target=consumer,args=(condition, image, test, ))
    c1.start()
    c1.join()
    response = {
        "status" : "Finished"
    }
    #model_maskRCNN.load_weights(os.path.join('server/static/model/mask_rcnn_type_0040.h5'), by_name=True)
    #results = model_maskRCNN.detect([image])
    return response
