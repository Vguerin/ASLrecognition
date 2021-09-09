#!usr/bin/python

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras import models
from collections import deque
from data_preprocessing import prepare_image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#Load trained model
my_model = models.load_model('model/DenseNet201_120_rgb_full_da_full_layers_add.h5')

class_id = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

cap = cv2.VideoCapture(0)
width_target = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_target = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Create a queue for mean averaging result
mean_result = np.zeros(len(class_id))
result_pred = deque(maxlen=15)
extended_max_x = deque(maxlen=5)
extended_max_y = deque(maxlen=5)
extended_min_x = deque(maxlen=5)
extended_min_y = deque(maxlen=5)

with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:

    while(True):

        # Capture frame-by-frame
        grabbed, img = cap.read()

        if not grabbed:
            print("[ERROR]: Problem with camera")
            break

        image_height = img.shape[0]
        image_width = img.shape[1]

        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # If no landmarks have been found, just display current frame
        if results.multi_hand_landmarks:

            # Find min and max coordinates from all landmark
            min_y = image_height+1
            max_y = 0
            min_x = image_width+1
            max_x = 0

            # Select only first landmark because we only show one hand
            for landmark in results.multi_hand_landmarks[0].landmark:

                # Landmark variable is normalize between 0 and 1, so
                # we need to scale it to image dimension
                scale_x = landmark.x*image_width
                scale_y = landmark.y*image_height

                if scale_x<min_x:
                    min_x=scale_x
                    
                if scale_x>max_x:
                    max_x=scale_x
                    
                if scale_y<min_y:
                    min_y=scale_y
                    
                if scale_y>max_y:
                    max_y=scale_y
             
            # Extend coordinates to get a proper hand image              
            extended_max_x.append(int(1.2*max_x) if (1.2*max_x)<image_width else image_width)
            extended_max_y.append(int(1.2*max_y) if (1.2*max_y)<image_height else image_height)
            extended_min_x.append(int(min_x/1.2) if (min_x/1.2)>0 else 0)
            extended_min_y.append(int(min_y/1.2) if (min_y/1.2)>0 else 0)

            # Average coordinate on few frames in order to avoid wavering
            avg_x_max = int(np.array(extended_max_x).mean(axis=0))
            avg_x_min = int(np.array(extended_min_x).mean(axis=0))
            avg_y_max = int(np.array(extended_max_y).mean(axis=0))
            avg_y_min = int(np.array(extended_min_y).mean(axis=0))

                
            selected_part = img[avg_y_min:avg_y_max, avg_x_min:avg_x_max]

            # Preprocessing image
            selected_part = prepare_image(selected_part)
            selected_part = np.expand_dims(selected_part,axis=0)

            prediction = my_model.predict(selected_part)

            # Predicion averaging on last predictions to smooth our result
            result_pred.append(prediction)
            mean_result = np.array(result_pred).mean(axis=0)
            index= np.argmax(mean_result)
            predict_class = class_id[index]

            text = "Prediction is : {} ({:.2f}%)".format(predict_class,mean_result[0][index]*100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img,(avg_x_min,avg_y_min), (avg_x_max,avg_y_max),(255, 0, 0),2)

            y_text = avg_y_min-10
            x_text = avg_x_min+10

            cv2.putText(img,text,(x_text,y_text), font, 0.65,(255,0,0),2,cv2.LINE_AA)
            cv2.imshow('American Sign Recognation',img)

        else:
            cv2.imshow('American Sign Recognation',img)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()