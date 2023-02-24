#!/usr/bin/env python
# coding: utf-8




import tensorflow as tf
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from imutils import paths
import cv2
import pickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG19, MobileNet
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from sklearn import svm, tree





# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())





LABELS = set(["Pass", "Fail", "#"])
 
# The list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
#
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
 

for imagePath in imagePaths:
	
	label = imagePath.split(os.path.sep)[-2]
 
	
	
	if label not in LABELS:
		continue
 
	
	#Muunnetaan RGB:ksi, koko 224x224
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
 
	#
	#kasvatetaan data ja labelit listaa
	data.append(image)
	labels.append(label)







#Data ja labelit numpy taulukoiksi
data = np.array(data)
labels = np.array(labels)

#Labelien encoodaus 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


#Jaetaan dataa train/test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)





# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects

mean = np.array([0.495, 0.476, 0.449], dtype="float32")
trainAug.mean = mean
valAug.mean = mean





baseModel = MobileNet(include_top=False,
	input_shape=(224, 224, 3)) 


# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# place the head FC model on top of the base model 

model = Model(inputs=baseModel.input, outputs=headModel)

# compile model
#print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])




#print("[INFO] training ...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])





# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))





# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])




# serialize the model to disk
#print("[INFO] serializing network...")
model.save(args["model"])
 
# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

