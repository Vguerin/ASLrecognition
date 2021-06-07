#!usr/bin/python

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from generate_dataset import get_dataset

def prepare_image(image, target_width = 224, target_height = 224):
    image_resized = cv2.resize(image,(target_width,target_height))
    return image_resized

my_model = tf.keras.models.load_model('asl_model.h5')

class_id = ['A',
'B',
'C',
'D',
'E',
'F',
'G',
'H',
'I',
'J',
'K',
'L',
'M',
'N',
'O',
'P',
'Q',
'R',
'S',
'space',
'T',
'U',
'V',
'W',
'X',
'Y',
'Z']

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = prepare_image(frame)
    width = img.shape[0]
    height = img.shape[1]


    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))

    hull = cv2.convexHull(contours)


    if len(hull)!=0:
        min_y = 10000
        max_y = 0
        min_x = 10000
        max_x = 0
        for element in hull:
            if element[0][0]<min_x:
                min_x=element[0][0]
                
            if element[0][0]>max_x:
                max_x=element[0][0]
                
            if element[0][1]<min_y:
                min_y=element[0][1]
                
            if element[0][1]>max_y:
                max_y=element[0][1]
                
        cv2.rectangle(img,(max_x+5,max_y+5),(min_x-5,min_y-5),(255, 0, 0),2)
        cv2.imshow('final',img)

        if (min_y-5)>0 and (min_x-5)>0 and (max_y+5)<=width and (max_x+5)<=height:
            selected_part = img[min_y-5:max_y+5, min_x-5:max_x+5]

        else:
            selected_part = img[min_y:max_y, min_x:max_x]

        if len(selected_part)!=0:
            selected_part = prepare_image(selected_part)
            selected_part = np.expand_dims(selected_part,axis=0)
            prediction = my_model.predict(selected_part)

            index= np.argmax(prediction)
            predict_class = class_id[index]

            text = "prediction is : " + predict_class
            print(text)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img,text,(width/2,height-20), font, .5,(255,255,255),2,cv2.LINE_AA)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()