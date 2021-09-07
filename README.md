# ASLrecognition

This project have been developed in order to use Tensorflow and Keras to design and develop deep learning model for American Sign Language recognition.

Here was implemented simple CNN architecture with few tuning parameters in order to get best accuracy model.

I decided to only use deep learning for sign recognition and use classic computer vision algorithm to detect hands in video.

Datasets with only hand sign were easier to find compare to hand sign with bounding box coordinates.


## Content

	1. Installation
	2. Input Data for training
	3. Data Preprocessing
	4. Training (with or without GCP)
	5. Results

## Installation

#TO DO: Faire un pip freeze pour savoir ma config

## Input Data

Open-Source Dataset have been used to deploy this solution. Huge thanks to https://www.kaggle.com/datamunge/sign-language-mnist, https://www.kaggle.com/grassknoted/asl-alphabet?select=asl_alphabet_train and https://www.kaggle.com/danrasband/asl-alphabet-test for there contributions.

I also record a video doing all sign language and export this video as frame with extract_img_from_video.py

Only static sign have been used for this case (J and Z haven't been taken).

Data have been splitted into training/testing/validation (70% - 15% - 15%) using split_data_training_test.py

Folder structures corresponding to sign language alphabet : from A to W (without J and Z) therefore we have 24 folders.

/Dataset
----> /A
----> /B
----> /C
...
----> /W

From this structure we create our /training, /testing and /validation folders

## Data Preprocessing

In order to get a robust model, I gather multiple dataset from different configuration but also add some data augmentation.

Indeed some sign could be seen from different angles therefore some zoom, width and height range have been added.

Horizontal flip has been added as well in order to interpret sign from left and right hand.

#TO DO: Ajouter des photos des data augmented
#TO DO: Expliquer le filtrage des données de certains dataset car trop redondant et donnaient de mauvais résultats sur de nouvelles data


I've also recorded a video doing all alphabet american sign language and extract every pictures to get more relevant data from my configuration.


## Training

Transfer learning have been used to generalize as much as possible our model. 

DenseNet have been chosen for this architecture because it present some advantages compared to the others : 

	- With its denseBlock and transition layers, it's strengthened feature propagation, allowed feature reutilisation and reduce vanishing gradient problem.
	- Reduce numbers of parameters
	- Training faster then ResNet for equal deepness
	- Complex architecture allowing to catch all characteristic from all sign language


#TO DO: Afficher différentes features map (au début et à la fin peut-être ?)


## Result

#TO DO: Ajouter une vidéo 


## Conclusion

Things to improve :

		- Use tf.data to improve training speed and have more control of data augmentation
		- Get more data from different genders and skin colors to robustify algorithm
		- Going more into data and clean it (might have some wrong image)
		




