import os

import PIL 
from PIL import Image
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
import cv2
import scipy



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


def get_parallels(lines, widths, angles, min_angle = 5) :
    kernel = scipy.stats.gaussian_kde(angles)
    
    mode = angles[np.argmax(kernel(angles))]
    expected = np.sum(angles*kernel.evaluate(angles))
    mean, std = np.mean(angles), np.std(angles)
    #print angles
    #print kernel.evaluate(angles)
    #print mean, expected
    
    if min_angle is None :
        parallel = np.where(np.abs(angles - mode) <= std)
    else : 
        parallel = np.where(np.abs(angles - mode) <= min_angle)
    return lines[parallel], widths[parallel], angles[parallel]


def get_tick_distance(img, lines, width) : 
    kernal_width = scipy.stats.gaussian_kde(width)
    #plt.title("Length")
    #sns.distplot(width)
    #plt.show()
    
    x = np.arange(img.shape[0])
    maximums = scipy.signal.argrelextrema(kernal_width(x), np.greater)
    minimums = scipy.signal.argrelextrema(kernal_width(x), np.less)
    #print maximums
    #print minimums
    t = len(minimums[0])
    lower = 0
    upper = img.shape[0]
    cur_color = 0
    dists = []
    
    ret = np.copy(img)
    for i in range(t) :
        indexes = np.logical_and(width >= lower, width < minimums[0][i])
        cur_lines = lines[indexes]
        if len(cur_lines) < 2 :
            continue
        
        
        ret, cur_dist = paint_ticks_and_calculate_dist(ret, cur_lines, cur_color)
        cur_color += 1
        dists.append(cur_dist)
        
        lower = minimums[0][i]
        
    indexes = width >= lower
    cur_lines = lines[indexes]
    if len(cur_lines) >= 2 :
        #sns.distplot(dists)
        ret, cur_dist = paint_ticks_and_calculate_dist(ret, cur_lines, cur_color)
        dists.append(cur_dist)

    return ret, dists

def paint_ticks_and_calculate_dist(img, lines, cur_color) :

    dists = get_lines_distance(lines)
    print(len(dists))
    if len(dists) == 1 :
        cur_dist = dists[0]
    else :
        kernel_dists = scipy.stats.gaussian_kde(dists)

        mode = dists[np.argmax(kernel_dists(dists))]
        cur_dist = mode
    
    cur_dist = int(cur_dist)
    
    pen_width = 1+ img.shape[0] // 100
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    use_color = colors[cur_color%6]
    
    ruler_height = np.max([img.shape[0] // 10, 10])
    #print ruler_height
    ruler = np.full((ruler_height, img.shape[1],3), 255).astype(np.uint8)

    cv2.line(ruler, (10, 0), (10, ruler_height), use_color, pen_width)
    cv2.line(ruler, (10+cur_dist, 0), (10+cur_dist, ruler_height), use_color, pen_width)
    cv2.line(ruler, (10, ruler_height/2), (10+cur_dist, ruler_height/2), use_color, pen_width)

    
    ret = np.copy(img)
    ret = np.vstack([ret, ruler])
    ret = ret.astype(np.uint8)
    
    for p in lines :
        #print p[:2].astype(int), p[2:].astype(int), colors[cur_color]
        cv2.line(ret, tuple(p[:2].astype(int)), tuple(p[2:].astype(int)), use_color, pen_width)
    return ret, cur_dist

def get_lines_distance(lines) : 
    t = len(lines)
    dist = np.full(t, np.inf)
    
    for i in range(t) :
        p = lines[i]
        for j in range(i+1,t) :
            p1 = lines[j]
            d = distance_point_line(p[:2], p1)
            if d < 0 :
                dist[j] = np.min((dist[j], np.abs(d)))
            else :
                dist[i] = np.min((dist[i], np.abs(d)))
    dist = dist[dist != np.inf]
    return dist

def distance_point_line(point, line) :
    dx = float(line[0] - line[2])
    dy = float(line[1] - line[3])

    if dx == 0 :
        return line[0] - point[0]

    m = dy/dx
    b = line[1] - m*line[0]

    y_dash = m*point[0] + b
    
    is_left = (y_dash < point[1])^ (m < 0) 

    d = np.abs(m*point[0]-point[1]+b)/np.sqrt(np.square(m)+1)
    if not is_left :
        d *= -1
    return d


def get_ticks(edge) :
    lsd = cv2.createLineSegmentDetector(0)

    #Detect lines in the image
    lines = lsd.detect(edge) #Position 0 of the returned tuple are the detected lines
    
    
    
    #Getting angles in degree
    dx = lines[0][:,0,0] - lines[0][:,0,2]
    dy = lines[0][:,0,1] - lines[0][:,0,3]
    angles = np.rad2deg(np.arctan(np.divide(dy,dx))) + 180
    angles %= 180
    
    #Getting length
    width = np.sqrt(np.square(dx) + np.square(dy))
    
    lines = lines[0]
    
    mean, std = np.mean(angles), np.std(angles)
    #sns.distplot(angles)
    #plt.show()
    ret = np.ones(edge.shape)
    
    for i in range(len(lines)):
        if np.abs(angles[i] - mean) > 3*std :
            continue
        #print angles[i]
        p = lines[i]
    #   print p
        cv2.line(ret, (int(p[0][0]),int(p[0][1])), (int(p[0][2]),int(p[0][3])), (0,0,0), 1)
    #   break
    #plt.figure(figsize= (20,5))
    #plt.imshow(ret)
    #plt.show()
    
    #print lines
    return lines[:,0], width, angles
    #Draw detected lines in the image

model_maskRCNN.load_weights(os.path.join(os.getcwd(),"../../type20200227T1718/mask_rcnn_type_0040.h5"), by_name=True)
image =  Image.open('/home/evida/Documents/Sensorbox_V2/sensor-box/server/static/images/tests/image2.jpg')
width, height = image.size
image2 = image.resize((int(width/4), int(height/4)), PIL.Image.ANTIALIAS)
results = model_maskRCNN.detect([np.array(image2)])
r = results[0]

index_ruler = [] 
for i in range(0, len(r['class_ids'])) : 
    if r['class_ids'][i] == 2 :
        index_ruler.append(i)
masks_Ruler  =np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
for i in range(len(index_ruler)):
    masks_Ruler = masks_Ruler +r['masks'][:,:,index_ruler[i]]

masks_Ruler[masks_Ruler >=1 ] = 1

kernel = np.ones((10,10), np.uint8)

ruler_dilation = cv2.dilate(masks_Ruler, kernel, iterations=1)
test_ruler = imageio.imsave('test_ruler.png',ruler_dilation)
print(np.array(image2).shape)
print(np.array(ruler_dilation).shape)
#only_Ruler = cv2.bitwise_and(np.array(image2), mask= np.array(ruler_dilation))
only_Ruler = np.array(image2).copy()
only_Ruler[np.array(ruler_dilation) == 0,:] = 0
only_Ruler[np.array(ruler_dilation) != 0,:] = np.array(image2)[np.array(ruler_dilation) != 0]


test_ruler = imageio.imsave('test_ruler.png',only_Ruler) 


lsd = cv2.createLineSegmentDetector(0)

#Detect lines in the image

lines, width, angles = get_ticks(cv2.cvtColor(np.array(only_Ruler), cv2.COLOR_BGR2GRAY))

lines, width, angles = get_parallels(lines, width, angles, min_angle=5)
print(lines)
#ticks_img, dists = get_tick_distance(np.array(only_Ruler), lines, width)
#print(ticks_img, dists)
'''
new_img, ratio = resize_ruler_img(img)
new_gt, ratio = resize_ruler_img(gt)

#ret = treat_colors(new_img, new_gt, remove_colors)
ret = treat_colors(new_img, new_gt, False)

#ret = binarize_img(ret, new_gt)
lines, width, angles = get_ticks(ret)
lines, width, angles = get_parallels(lines, width, angles, min_angle=10)
lines, width, angles = merge_lines(lines, width, min_h=2)

lines_original = lines/ratio
width_original = width/ratio
ticks_img, dists = get_tick_distance(img, lines_original, width_original)
ticks_img
dists
'''