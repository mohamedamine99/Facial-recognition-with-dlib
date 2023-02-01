# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:48:02 2023

@author: ASUS
"""

from  my_dlib_funcs import *
import random

dlib.DLIB_USE_CUDA = False

print(os.getcwd())

os.chdir('C:/Users/ASUS/Desktop/Coursera DL/spyder projects/real-time facial recognition')

print(os.getcwd())


imgs_path = os.getcwd() + '/imgs'
database_path =  os.getcwd() + '/database'
test_path = os.getcwd() + '/testing imgs'
cnn_model_path = os.getcwd() + '/mmod_human_face_detector.dat'
shape_predictor_path = os.getcwd() + '/shape_predictor_68_face_landmarks_GTX.dat'
face_recognition_model_path = os.getcwd() + "/dlib_face_recognition_resnet_model_v1.dat"
output_path = os.getcwd() +'/Outputs'



cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_recognition_model_path)





db_descriptors = get_descriptors(imgs_path =database_path , face_detector = cnn_face_detector, shape_predictor = predictor, face_recognizer = face_rec )

test_descriptors = get_descriptors(imgs_path =test_path,face_detector = cnn_face_detector, shape_predictor = predictor, face_recognizer = face_rec )

        
recognize(db_descriptors,test_descriptors)


display(test_descriptors,save_output = True, save_path = output_path)