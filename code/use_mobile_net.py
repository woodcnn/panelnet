#!/usr/bin/env python
# coding: utf-8

#The original code is made by Adrian Rosebrock
#https://pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/


# import the necessary packages
import tensorflow as tf
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report





ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
#ap.add_argument("-i", "--input", required=True,
	#help="path to our input video")
ap.add_argument("-o", "--output", required=False,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())





# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())





# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([0.495, 0.476, 0.449][::1], dtype="float32")
Q = deque(maxlen=args["size"])


# In[4]:


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)
#vs = cv2.VideoCapture(args["input"])


writer = None
(W, H) = (None, None)


# In[5]:

#(H, W) = (640, 480) #frame.shape[:2]
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
    
 
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
 
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
        
        	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean
    
   	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	

	probs = model.predict(np.expand_dims(frame, axis=0)) 
	probs2 = np.round(probs, 3) 
	
	probs2 = probs2 * 100
 	



	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	
	i = np.argmax(results)
	label = lb.classes_[i]
    
    	# draw the activity on the output frame
	text = "quality: {}".format(label)
	text2 = "{}".format(probs2)
	
	#cv2.putText(output, text,  (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
		#1.25, (0, 255, 0), 5)
 
	#cv2.putText(output, text2,  (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
		#0.65, (0, 255, 0), 2)
	
# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 5,
			(W, H), True)
 
	# write the output frame to disk
	writer.write(output)
 	
	 #show the output image
	output = cv2. resize(output, (640, 480)) 	

	cv2.imshow("Output", output)
	
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
        
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

