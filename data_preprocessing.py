import cv2
import numpy as np

def prepare_image(image, target_width = 120, target_height = 120):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalization
    image = image * (1.0/255.0)
    # calculate global mean and standard deviation
    image_asarray = np.array(image)
    mean, std = image_asarray.mean(), image_asarray.std()
    # global standardization
    image = (image - mean) / std
    image_resized = cv2.resize(image,(target_width,target_height))
    return image_resized