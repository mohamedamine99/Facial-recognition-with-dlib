# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:18:13 2023

@author: ME

This module provides convenient functions for utilizing dlib in facial recognition tasks.

"""

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

dlib.DLIB_USE_CUDA = False




def calculate_distance(descriptor1 = None ,descriptor2 = None):
    return np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))


def get_descriptors(imgs_path ='' , face_detector = None, shape_predictor = None, face_recognizer = None , jitter = 1 ):
    face_descriptors = []
    for f in os.listdir(imgs_path):
        img = dlib.load_rgb_image(imgs_path +'/'+ f)

        faces = face_detector(img)
    
        print("Number of faces detected: {}".format(len(faces)))
        for i, d in enumerate(faces):   
            cache = {}
            shape = shape_predictor(img, d.rect)
            face_descriptor = face_recognizer.compute_face_descriptor(img, shape, jitter)                       
            img_path = imgs_path +'/'+ f
            img_name = f[:-4]
            
            cache["face descriptor"] = face_descriptor
            cache["img name"] = img_name
            cache["img path"] = img_path
            cache["bounding box"] = d.rect
            
            
            face_descriptors.append(cache)
            
    return face_descriptors


def recognize(db_descriptors = None ,test_descriptors = None, distance_thresh = 0.55):
    for test_face in test_descriptors:
        distances = []
        for db_face in db_descriptors:

            dist = calculate_distance(np.array(test_face["face descriptor"]) , np.array(db_face["face descriptor"]))
            distances.append(dist)
            
        idx = np.argmin(distances)
        print(distances , idx)
        if distances[idx] > distance_thresh:
            test_face["img name"] = "UNKNOWN"
            print(idx)
        else:
            test_face["img name"] = db_descriptors[idx]["img name"]
            
            
            
def display(test_descriptors = None, rows = 2 , save_output = False ,save_path = ''):
    
    fig = plt.figure(figsize=(15, 7))
    
    total_imgs = len(test_descriptors)
    if total_imgs % rows == 0:
        columns = total_imgs // rows
    else :
        columns = total_imgs // rows + 1
    
    for j, test_face in enumerate(test_descriptors):
        
        img = dlib.load_rgb_image(test_face["img path"])
        left = test_face["bounding box"].left()
        top = test_face["bounding box"].top()
        right = test_face["bounding box"].right()
        bottom = test_face["bounding box"].bottom()
        
        img = cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),thickness = 4)
      
        fig.add_subplot(rows, columns, j+1)
            
            # showing image
        plt.imshow(img)
        plt.axis('off')
        plt.title(test_face["img name"])
        if save_output:
            img_with_text = cv2.putText(img,test_face["img name"],(left,top),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0,0,0),thickness = 2)
            
            img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)                        
            cv2.imwrite(save_path +'/'+ test_face["img name"] + str(j+1) + ".png",
                        img_with_text)

        
    if save_output:
        plt.savefig(save_path +'/' + 'output_fig.png')
