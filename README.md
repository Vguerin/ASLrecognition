# ASLrecognition
<p align="center">
<img src="https://user-images.githubusercontent.com/64073885/132715154-11c91247-d031-4755-a53b-b2927b31bd9e.png" width="400" height="500">
</p>
This project have been developed in order to use Tensorflow and Keras to design and develop deep learning model for American Sign Language recognition.

Here was implemented CNN architecture with few tuning parameters in order to get best accuracy model.

I decided to only use deep learning for sign recognition and use computer vision algorithm for hands detection in video.

Datasets with only hand sign were easier to find compare to hand sign with bounding box coordinates.


## Content

	1. Installation
	2. Input Data for training
	3. Data Preprocessing
	4. Training (with or without GCP)
	5. Results

## 1. Installation

Python 3.8.5 needed

```
pip install tensorflow-gpu==2.5.0 opencv-python==4.4.0.46 numpy==1.19.5 matplotlib==3.3.3
```

## 2. Input Data

Open-Source Dataset have been used to deploy this solution. Huge thanks to https://www.kaggle.com/datamunge/sign-language-mnist, https://www.kaggle.com/grassknoted/asl-alphabet?select=asl_alphabet_train and https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset for there contributions.

I also record a video doing all sign language and export this video as frame with ***extract_img_from_video.py***

```
python split_data_training_test.py absolute_path_of_target_video
```

Only static sign have been used for this case (J and Z haven't been taken into account).

Data have been splitted into training/testing/validation (70% - 15% - 15%) using ***split_data_training_test.py***

```
python split_data_training_test.py absolute_path_of_target_folder
```

Folder structures corresponding to sign language alphabet : from A to W (without J and Z) therefore we have 24 folders.

- /Dataset
  - ----> /A
  - ----> /B
  - ----> /C
  - ----> ...
  - ----> /W

From this structure we create our **/training**, **/testing** and **/validation folders**

## 3. Data Preprocessing

In order to get a robust model, I gather multiple dataset from different configuration but also add some data augmentation.

Indeed some sign could be seen from different angles therefore some zoom, width and height range have been added.

Horizontal flip has been added as well in order to interpret sign from left and right hand.

<p align="center">
<img src="https://user-images.githubusercontent.com/64073885/132717017-5e0cead1-12fe-4b4a-b2b4-b1d4c576ef19.jpg">
</p>

![data_augmentation](https://user-images.githubusercontent.com/64073885/132717051-3ace955a-6a0d-4af7-9e75-8ac781364479.jpg)

I've also recorded a video doing all alphabet american sign language and extract every pictures to get more relevant data from my configuration.


## 4. Training

Transfer learning have been used to generalize as much as possible our model. 

DenseNet have been chosen for this architecture because it present some advantages compared to the others : 

- With its denseBlock and transition layers, it's strengthened feature propagation, allowed feature reutilisation and reduce vanishing gradient problem.
- Reduce numbers of parameters
- Training faster then ResNet for equal deepness

Here you can see features map from first convolutionnal layer the 69th convolutionnal layer of this architecture. 

<p align="center">
	<b>Output from 1st Conv Layer</b><br>
	<img src="https://user-images.githubusercontent.com/64073885/132724594-4cc2a156-a349-46af-a824-9b68012d435c.jpg" width="400" height="500">
</p>

<p align="center">
	<b>Output from 69th Conv Layer</b><br>
	<img src="https://user-images.githubusercontent.com/64073885/132724608-ac073a3f-4cbd-4aee-9c87-6064d0369b80.jpg" width="400" height="500">
</p>

Clearly we can see that going deeper into network we lose some informations but thanks to DenseNet and its denseBlock (which every layer receive all preceding layers), it still can detect some features.

## 5. Result


https://user-images.githubusercontent.com/64073885/132715333-1e09e560-5279-4f7f-ad14-260b86391852.mp4

To do hands detection, 2 methods have been tried :

- Using OpenCV and applying some color mask in HSV mode, then find contours, compute convex hull and find extreme coordinates
- Using mediapipe

I decided to finally use mediapipe because first method had tendance to detect also forearm which decreased my results

To use it, you can follow Mediapipe installation instructions here :  https://google.github.io/mediapipe/getting_started/install.html

## Conclusion

Things to improve :

		- Use tf.data to improve training speed and have more control of data augmentation
		- Get more data from different genders, size, skin colors and both hands to robustify algorithm
		- To go more into data cleaning (might have some wrong image)
		- To do more hyperparameters tuning 
		




