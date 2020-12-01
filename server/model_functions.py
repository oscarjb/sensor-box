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

import PIL 
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
        #print(r_g)
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
        
        

        print(" Measuring percentage of Granulation:  Start")
        number_g= 0
        number_upp=sum(sum(masks_UPP.astype(int)))
        
        print(number_upp)
        test.Pixels_in_UPP = number_upp

        print(sum(sum(masks_granulation.astype(int))))
        union = masks_UPP.astype(int) + masks_granulation.astype(int)
        zeross = np.zeros((union.shape[0],union.shape[1]))
        intersection = sum(sum(np.where(union==2 , union, zeross)))
        print(intersection)
        test.Pixels_in_g = intersection/2
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


        if (Percentage_g>0):
            test.TissueTypes =  test.TissueTypes + 'Granulation'
        
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





################################## Slough classification & percentage ################################################
        yellow = [1.0,1.0,0.0] # Yellow for Slough
        masked_s = image.copy()
        model_slough.load_weights(os.path.join(os.getcwd(),"../Slough_model/mask_rcnn_type_0050.h5"), by_name=True)
        results_s = model_slough.detect([image])
        r_s = results_s[0]

        print(" Detection of Slough:  Done")
        #print(r_s)
        masks_slough  =np.zeros((r_s['masks'].shape[0], r_s['masks'].shape[1]))
        for i in range(len(r_s['class_ids'])):
            masks_slough = masks_slough +r_s['masks'][:,:,i]

        masks_slough[masks_slough >=1 ] = 1
        for c in range(3):
            masked_s[:, :, c] = np.where(masks_slough.astype(int) == 1,
                                  masked_s[:, :, c] *
                                  (1 - alpha) + alpha * yellow[c]* 255,
                                  masked_s[:, :, c])

        print(" Segmentation of Slough:  Done")

        image_store_db_s = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_s.png', masked_s) 
        test.Segmented_leftImage_s = 'tests/' + pathnn + '_segmented_s.png'
        print(" Storing segmented Slough:  Done")
        
        
        
        

        print(" Measuring percentage of Slough:  Start")
        number_s= 0
        number_upp=sum(sum(masks_UPP.astype(int)))
        
        union_s = masks_UPP.astype(int) + masks_slough.astype(int)
        zeross_s = np.zeros((union_s.shape[0],union_s.shape[1]))
        intersection_s = sum(sum(np.where(union_s==2 , union_s, zeross_s)))
        print(intersection_s)
        test.Pixels_in_s = intersection_s/2
        if number_upp!=0:            
            Percentage_s = intersection_s *100 / (2*number_upp)
        else: 
            Percentage_s = 0
        
        print(Percentage_s)

        if (Percentage_s>0):
            test.TissueTypes =  test.TissueTypes + ' Slough'
        
        
        test.Slough =  str("{:.2f}".format(Percentage_s)) + ' %'
             
        test_s = imageio.imsave('test_s.png', np.where(union_s==2 , union_s, zeross_s).astype(int)) 

               
        print(" Measuring percentage of Slough:  Done")



################################## Necrosis classification & percentage ################################################
        purple = [0.15,0.0,0.4] # Purple for Necrosis
        masked_n = image.copy()
        model_necrosis.load_weights(os.path.join(os.getcwd(),"../Necrosis_model/mask_rcnn_type_0040.h5"), by_name=True)
        results_n = model_necrosis.detect([image])
        r_n = results_n[0]

        print(" Detection of Necrosis:  Done")
        #print(r_s)
        masks_necrosis  =np.zeros((r_n['masks'].shape[0], r_n['masks'].shape[1]))
        for i in range(len(r_n['class_ids'])):
            masks_necrosis = masks_necrosis +r_n['masks'][:,:,i]

        masks_necrosis[masks_necrosis >=1 ] = 1
        for c in range(3):
            masked_n[:, :, c] = np.where(masks_necrosis.astype(int) == 1,
                                  masked_n[:, :, c] *
                                  (1 - alpha) + alpha * purple[c]* 255,
                                  masked_n[:, :, c])

        print(" Segmentation of Necrosis:  Done")

        image_store_db_n = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_n.png', masked_n) 
        test.Segmented_leftImage_n = 'tests/' + pathnn + '_segmented_n.png'
        print(" Storing segmented Necrosis:  Done")
        
        
        
        

        print(" Measuring percentage of Necrosis:  Start")
        number_n= 0
        number_upp=sum(sum(masks_UPP.astype(int)))
        
        union_n = masks_UPP.astype(int) + masks_necrosis.astype(int)
        zeross_n = np.zeros((union_n.shape[0],union_n.shape[1]))
        intersection_n = sum(sum(np.where(union_n==2 , union_n, zeross_n)))
        
        print(intersection_n)
        
        if number_upp!=0:            
            Percentage_n = intersection_n *100 / (2*number_upp)
        else: 
            Percentage_n = 0
        
        print(Percentage_n)

        if (Percentage_n>0):
            test.TissueTypes =  test.TissueTypes + ' Necrosis'
        
        
        test.Necrosis =  str("{:.2f}".format(Percentage_n)) + ' %'
             
        test_n = imageio.imsave('test_n.png', np.where(union_n==2 , union_n, zeross_n).astype(int)) 

               
        print(" Measuring percentage of Necrosis:  Done")


        ################################## Saving the patient data ################################################
        if (test.TissueTypes == ''):
            test.TissueTypes = 'None'
        
        test.save()
        #print(r['class_ids'])
        cond.wait(0.1)
        logging.debug('Resource is available to consumer')



def consumer2(cond, path_upp, path_tissues,path_distance,test):
    """wait for the condition and use the resource"""
    logging.debug('Starting consumer thread')
    with cond:

        image2 = Image.open(os.path.join('server/static/images/', str(test.camera_kurokesu).replace('\\', '/') ))
        print("image2", 'server/static/images/', str(test.camera_kurokesu).replace('\\', '/') )
        image2 = np.array(image2.resize((640,360), PIL.Image.ANTIALIAS))
        if(os.path.exists(os.path.join('server/static/images/edited/' + path_upp ))):
            print("Edited ULCER Exists")
            image_edited_ulcer = Image.open((os.path.join('server/static/images/edited/' + path_upp )))
            image_edited_ulcer = np.array(image_edited_ulcer.resize((640, 360), PIL.Image.ANTIALIAS))
            mask_upp=np.zeros([640, 360])
            red = [1.0,0.0,0.0] # Red for UPP 
            alpha=0.5
            reds=np.array(image_edited_ulcer)[:,:,0]
            greens=np.array(image_edited_ulcer)[:,:,1]
            blues=np.array(image_edited_ulcer)[:,:,2]
            mask_upp = (reds == 0) & (greens ==0 ) &  (blues ==255)
            mask_ulcer = np.where(mask_upp, 1, 0)
            sum_of_pixels_in_UPP = sum(sum(mask_ulcer.astype(int)))
            test.Pixels_in_UPP = sum_of_pixels_in_UPP
            masked_UPP = image2.copy()
            for c in range(3):
                masked_UPP[:, :, c] = np.where(mask_upp.astype(int) == 1,
                                  masked_UPP[:, :, c] *
                                  (1 - alpha) + alpha * red[c]* 255,
                                  masked_UPP[:, :, c])

            print(" Segmentation of edited upp :  Done")
            pathn = str(test.camera_kurokesu)
            start = pathn.find('test') + 6
            end = pathn.find('.jpg', start)
            pathnn = pathn[start:end]
            image_store_db = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_edited.png', masked_UPP) 
            test.Segmented_leftImage = 'tests/' + pathnn + '_segmented_edited.png'


        else:
            image_edited_ulcer = Image.open((os.path.join('server/static/images/edited/' + path_upp )))
            image_edited_ulcer = np.array(image_edited_ulcer.resize((640, 360), PIL.Image.ANTIALIAS))
            print("Edited ULCER DOES NOT Exist")
            #sum_of_pixels_in_UPP = test.Pixels_in_UPP
            model_maskRCNN.load_weights(os.path.join(os.getcwd(),"../type20200227T1718/mask_rcnn_type_0040.h5"), by_name=True)

            results = model_maskRCNN.detect([image_edited_ulcer])
            r = results[0]
            
            masked_image = image_edited_ulcer.copy()
            index_upp = [] 
            for i in range(0, len(r['class_ids'])) : 
                if r['class_ids'][i] == 1 : 
                    index_upp.append(i)
            mask_upp = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
            
            for i in range(len(index_upp)):
                mask_pp = mask_upp +r['masks'][:,:,index_upp[i]]
            mask_upp[mask_upp >=1 ] = 1



        if(os.path.exists(os.path.join('server/static/images/edited/' + path_tissues ))):
            print("Edited Tissues Exists")
            test.TissueTypes = ''
            red= [1.0,0.0,0.0] # red for granulation
            yellow = [1.0,1.0,0.0] # Yellow for Slough
            purple = [0.15,0.0,0.4] # Purple for Necrosis
            image_edited_ulcer = Image.open((os.path.join('server/static/images/edited/' + path_tissues )))
            image_edited_ulcer =  np.array(image_edited_ulcer.resize((640, 360), PIL.Image.ANTIALIAS))
            mask_g=np.zeros([640, 360])
            mask_n=np.zeros([640, 360])
            mask_s=np.zeros([640, 360])
            reds=np.array(image_edited_ulcer)[:,:,0]
            greens=np.array(image_edited_ulcer)[:,:,1]
            blues=np.array(image_edited_ulcer)[:,:,2]
            mask_g = (reds == 255) & (greens ==0 ) &  (blues ==0)
            mask_s = (reds == 255) & (greens ==255 ) &  (blues ==0)
            mask_n = (reds == 128) & (greens ==0 ) &  (blues ==128)
            mask_granulation = np.where(mask_g, 1, 0)
            mask_slough = np.where(mask_s, 1, 0)
            mask_necrosis = np.where(mask_n, 1, 0)
            sum_of_pixels_in_g = sum(sum(mask_ulcer.astype(int)))
            sum_of_pixels_in_s = sum(sum(mask_ulcer.astype(int)))
            sum_of_pixels_in_n = sum(sum(mask_ulcer.astype(int)))

            test.Pixels_in_g = sum_of_pixels_in_g
            test.Pixels_in_s = sum_of_pixels_in_s
            test.Pixels_in_n = sum_of_pixels_in_n

            union_g = mask_upp.astype(int) + mask_granulation.astype(int)
            union_s = mask_upp.astype(int) + mask_slough.astype(int)
            union_n = mask_upp.astype(int) + mask_necrosis.astype(int)
            zeross_g = np.zeros((union_g.shape[0],union_g.shape[1]))
            zeross_s = np.zeros((union_s.shape[0],union_s.shape[1]))
            zeross_n = np.zeros((union_n.shape[0],union_n.shape[1]))
            intersection_n = sum(sum(np.where(union_g==2 , union_g, zeross_g)))
            intersection_g = sum(sum(np.where(union_s==2 , union_s, zeross_s)))
            intersection_s = sum(sum(np.where(union_n==2 , union_n, zeross_n)))
            
            print(intersection_g)
            print(intersection_s)
            print(intersection_n)
            
            if test.Pixels_in_UPP!=0:            
                Percentage_g = intersection_g *100 / (2*test.Pixels_in_UPP)
                Percentage_s = intersection_s *100 / (2*test.Pixels_in_UPP)
                Percentage_n = intersection_n *100 / (2*test.Pixels_in_UPP)
            else: 
                Percentage_g = 0
                Percentage_s = 0
                Percentage_n = 0
            test.Granulation =  str("{:.2f}".format(Percentage_g)) + ' %'
            test.Slough =  str("{:.2f}".format(Percentage_s)) + ' %'
            test.Necrosis =  str("{:.2f}".format(Percentage_n)) + ' %'
            print(Percentage_g)
            print(Percentage_s)
            print(Percentage_n)


            if (Percentage_g>0):
                test.TissueTypes =  test.TissueTypes + ' Granulation'
            if (Percentage_s>0):
                test.TissueTypes =  test.TissueTypes + ' Slough'
            if (Percentage_n>0):
                test.TissueTypes =  test.TissueTypes + ' Necrosis'


            masked_Granulation = image2.copy()
            masked_Slough = image2.copy()
            masked_Necrosis = image2.copy()
            for c in range(3):
                masked_Granulation[:, :, c] = np.where(mask_granulation.astype(int) == 1,
                                  masked_Granulation[:, :, c] *
                                  (1 - alpha) + alpha * red[c]* 255,
                                  masked_Granulation[:, :, c])
                masked_Slough[:, :, c] = np.where(mask_slough.astype(int) == 1,
                                  masked_Slough[:, :, c] *
                                  (1 - alpha) + alpha * yellow[c]* 255,
                                  masked_Slough[:, :, c])
                masked_Necrosis[:, :, c] = np.where(mask_necrosis.astype(int) == 1,
                                  masked_Necrosis[:, :, c] *
                                  (1 - alpha) + alpha * purple[c]* 255,
                                  masked_Necrosis[:, :, c])

            print(" Segmentation of edited tissues :  Done")
            pathn = str(test.camera_kurokesu)
            start = pathn.find('test') + 6
            end = pathn.find('.jpg', start)
            pathnn = pathn[start:end]
            image_store_db_g = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_g_edited.png', masked_Granulation) 
            test.Segmented_leftImage_g = 'tests/' + pathnn + '_segmented_g_edited.png'
            image_store_db_s = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_s_edited.png', masked_Slough) 
            test.Segmented_leftImage_s = 'tests/' + pathnn + '_segmented_s_edited.png'
            image_store_db_n = imageio.imsave('server/static/images/tests/'+ pathnn + '_segmented_n_edited.png', masked_Necrosis) 
            test.Segmented_leftImage_n = 'tests/' + pathnn + '_segmented_n_edited.png'



        # else:
        #     sum_of_pixels_in_g = test.Pixels_in_g
        #     sum_of_pixels_in_s = test.Pixels_in_s
        #     sum_of_pixels_in_n = test.Pixels_in_n
        if (test.TissueTypes == ''):
            test.TissueTypes = 'None'
        
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

with tf.device(DEVICE):
    model_slough = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config_g)
with tf.device(DEVICE):
    model_necrosis = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                            config=config_g)



def segment_edited(path_upp,path_tissues, path_distance,test):
    condition = threading.Condition()
    c1 = threading.Thread(name='c1', target=consumer2,args=(condition, path_upp,path_tissues, path_distance, test, ))
    c1.start()
    c1.join()
    response = {
        "status" : "Finished"
    }
    #model_maskRCNN.load_weights(os.path.join('server/static/model/mask_rcnn_type_0040.h5'), by_name=True)
    #results = model_maskRCNN.detect([image])
    return response

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


