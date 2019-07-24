from __future__ import print_function
import os
from scipy import misc
import glob
import matplotlib.pyplot as plt   
import sys 
from scipy.ndimage import rotate
from scipy.misc import imread, imshow
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import sys
import shutil
import random
import numpy as np
from keras.models import load_model
import pandas as pd
import cv2
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
import os

path=r"G:\study\machine learning\competition\analytics Vidya Computer Vision"
train=pd.read_csv(path+"\\train.csv")
test=pd.read_csv(path+"\\test_ApKoW4T.csv")

train_size=5000
test_size=1252


def make_dir(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        
path_train = path+"\\train"
make_dir(path_train)
path_test = path+"\\test"
make_dir(path_test)


test_real=[]
train_real=[]
path_test_real = path+"\\test_real"
make_dir(path_test_real)
path_train_real = path+"\\train_real"
make_dir(path_train_real)


for i in range(0,len(test)):
    image = cv2.imread(path+"\\images\\"+test['image'][i]) 
    test_real.append(image)
    cv2.imwrite(os.path.join(path_test_real+"\\"+test['image'][i]),image)


for i in range(0,len(train)):
    image = cv2.imread(path+"\\images\\"+train['image'][i]) 
    train_real.append(image)
    cv2.imwrite(os.path.join(path_train_real+"\\"+train['image'][i]),image)

path1=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\training\1"        
path2=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\training\2"
path3=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\training\3"
path4=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\training\4"
path5=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\training\5"

for i in range(0,5000):
    if(train['category'][i]==1):
        cv2.imwrite(os.path.join(path1+"//"+train['image'][i]),train_real[i])
    elif(train['category'][i]==2):
        cv2.imwrite(os.path.join(path2+"//"+train['image'][i]),train_real[i])    
    elif(train['category'][i]==3):
        cv2.imwrite(os.path.join(path3+"//"+train['image'][i]),train_real[i])    
    elif(train['category'][i]==4):
        cv2.imwrite(os.path.join(path4+"//"+train['image'][i]),train_real[i])    
    else:
        cv2.imwrite(os.path.join(path5+"//"+train['image'][i]),train_real[i])    


path1=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\testing\1"        
path2=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\testing\2"
path3=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\testing\3"
path4=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\testing\4"
path5=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\testing\5"

for i in range(5000,len(train_real)):
    if(train['category'][i]==1):
        cv2.imwrite(os.path.join(path1+"//"+train['image'][i]),train_real[i])
    elif(train['category'][i]==2):
        cv2.imwrite(os.path.join(path2+"//"+train['image'][i]),train_real[i])    
    elif(train['category'][i]==3):
        cv2.imwrite(os.path.join(path3+"//"+train['image'][i]),train_real[i])    
    elif(train['category'][i]==4):
        cv2.imwrite(os.path.join(path4+"//"+train['image'][i]),train_real[i])    
    else:
        cv2.imwrite(os.path.join(path5+"//"+train['image'][i]),train_real[i])    





print("Training and Test Data Extraction Complete")

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
#script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(r'G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\training')
test_set_path = os.path.join(r'G:\study\machine learning\competition\analytics Vidya Computer Vision\dataset\testing')
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
input_size = (192, 192)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=5, activation='softmax'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='categorical')
 
test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='categorical')
 
# Create a loss history
history = LossHistory()
 
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=100,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         callbacks=[history])
 
 
# Save model
model_backup_path = os.path.join(r'G:\study\machine learning\competition\analytics Vidya Computer Vision\model3.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(r'G:\study\machine learning\competition\analytics Vidya Computer Vision\loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)

import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
images = []
path1=r"G:\study\machine learning\competition\analytics Vidya Computer Vision\test_real\\"
for img in os.listdir(r"G:\study\machine learning\competition\analytics Vidya Computer Vision\test_real"):
    img = image.load_img(path1+img, target_size=(192, 192))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)

classifier = tf.keras.models.load_model(r"G:\study\machine learning\competition\analytics Vidya Computer Vision\model3.h5")
classifier._make_predict_function()
classes=classifier.predict(images)

pred=[]
for i in range(0,len(classes)):
    pred.append(classes[i].argmax(axis=0)+1)

import pandas as pd
z=os.listdir(r"G:\study\machine learning\competition\analytics Vidya Computer Vision\test_real")
data=pd.DataFrame(list(zip(z,pred)),columns=['image','category']).to_csv(r"G:\study\machine learning\competition\analytics Vidya Computer Vision\Solution3.csv")




