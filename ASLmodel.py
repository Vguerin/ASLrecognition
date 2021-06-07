#!/usr/bin/python
import tensorflow as tf
import numpy as np


class InterBlock(tf.keras.layers.Layer):
	def __init__(self,input_neurons,nb_block,input_shape,**kwargs):
		super().__init__(**kwargs)
		self.hidden = [tf.keras.layers.Conv2D(input_neurons, (3, 3),activation='relu',input_shape=input_shape),
						tf.keras.layers.MaxPooling2D(2, 2),
						tf.keras.layers.Dropout(0.2)]
		for block in range(1,nb_block):
			temp = [tf.keras.layers.Conv2D((input_neurons/2), (3, 3), activation='relu'),
							tf.keras.layers.MaxPooling2D(2, 2),
							tf.keras.layers.Dropout(0.2)]
			self.hidden = np.hstack((self.hidden,temp))
		print(self.hidden)


class ASLmodel(tf.keras.Model):
	def __init__(self,output_dim,input_shape,**kwargs):
		super().__init__(**kwargs)
		self.hidden_layers = InterBlock(128,3,input_shape)
		self.block1 = tf.keras.layers.Flatten()
		self.block2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
		self.block3 = tf.keras.layers.Dropout(0.2)
		self.out = tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)


	def call(self,inputs):
		Z = self.hidden_layers(inputs)
		Z = self.block1(Z)
		Z = self.block2(Z)
		Z = self.block3(Z)
		return self.out(Z)


		






# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(28, activation=tf.nn.softmax)])

