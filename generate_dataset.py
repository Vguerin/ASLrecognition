#!usr/bin/python
import os
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from random import sample


def get_labels_id(path):

	classes = []

	data_dir = os.path.join(os.getcwd(),path)
	image_paths = defaultdict(list)

	#Get every folder names corresponding to a letter
	for subdir, dirs, files in os.walk(data_dir):
	    if subdir.split('\\')[-1] == 'dataset':
	        continue
	    folder = subdir.split('\\')[-1]
	    classes.append(folder)
	    
	    #Create image path for every images of every class
	    for file in files:
	        filepath = os.path.join(subdir, file)
	        if filepath.endswith(".jpg"): # or filepath.endswith(".png"):
	            image_paths[classes[-1]].append(filepath)

	img_class_ids = {classe: index for index, classe in enumerate(classes)}

	img_paths_and_classes = []
	for classe, paths in image_paths.items():
	    for path in paths:
	        img_paths_and_classes.append((path, img_class_ids[classe]))

	return img_class_ids,img_paths_and_classes

def prepare_image(image, target_width = 224, target_height = 224):
    
    image_resized_grey = cv2.resize(image,(target_width,target_height))
    image_resized = cv2.cvtColor(image_resized_grey,cv2.COLOR_BGR2RGB)
    
    # Let's also flip the image horizontally with 50% probability:
    if np.random.rand() < 0.5:
        image_resized = np.fliplr(image_resized)

    return image_resized.astype(np.float32)

def norm_between_a_b(array_img,a,b):
    img_norm = []
    for array in array_img:
        min_array = array.min()
        max_array = array.max()
        array_norm = a-(array-min_array)*(a-b)/(max_array-min_array)
        img_norm.append(array_norm)
    return img_norm

def prepare_all_images(img_paths_and_classes):
    images = [mpimg.imread(path)[:, :, :] for path, labels in img_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    prepared_images = norm_between_a_b(prepared_images,0,1)
    X_all = np.stack(prepared_images)
    Y_all = np.array([labels for path, labels in img_paths_and_classes], dtype=np.int32)
    return X_all, Y_all


def get_dataset(path):
	class_id,img_paths_and_classes = get_labels_id(path)
	test_ratio = 0.2
	train_size = int(len(img_paths_and_classes) * (1 - test_ratio))

	np.random.shuffle(img_paths_and_classes)

	img_paths_and_classes_train = img_paths_and_classes[:train_size]
	img_paths_and_classes_test = img_paths_and_classes[train_size:]

	validation_ratio = 0.3
	train_size = int(len(img_paths_and_classes_train) * (1 - validation_ratio))

	np.random.shuffle(img_paths_and_classes_train)

	img_paths_and_classes_training = img_paths_and_classes[:train_size]
	img_paths_and_classes_validation = img_paths_and_classes[train_size:]

	X_train, y_train = prepare_all_images(img_paths_and_classes_training)
	X_val, y_val = prepare_all_images(img_paths_and_classes_validation)
	X_test, y_test = prepare_all_images(img_paths_and_classes_test)

	return X_train,y_train,X_test,y_test,X_val,y_val,class_id

if __name__=='__main__':
	get_dataset('dataset')