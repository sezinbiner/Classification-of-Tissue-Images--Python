# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/gdrive')

os.chdir("/content") 
#transfer dataset from google drive
!unzip "/content/gdrive/My Drive/dataset/benign.zip"
!unzip "/content/gdrive/My Drive/dataset/InSitu.zip"
!unzip "/content/gdrive/My Drive/dataset/invasive.zip"
!unzip "/content/gdrive/My Drive/dataset/Normal.zip"

import os
import cv2
DATADIR = "/content" 
Categories = ["benign","InSitu","invasive","Normal"] 
angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

x = []
y = []
patch_size = 256                       #all images are resized to this size
for category in Categories: 
  path = os.path.join(DATADIR,category)
  print(path)
  os.chdir(path) 
  label = Categories.index(category)  #label 0->Benign 1-> InSitu 2->Invasive 3->Normal
  for im in os.listdir(path):
      try:
          img = cv2.imread(im,cv2.IMREAD_COLOR)           #cv2 reads images in BGR format
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #transform images into RGB format
          img = cv2.resize(img,(patch_size, patch_size))  #resize all images to same size 256*256
          """
          import matplotlib.pyplot as plt
          plt.imshow(img)
          plt.show()
          """
          x.append(img)                                    #in x images are stored
          y.append(label)                                  #in y labels are stored
          for angle in angles:                             #images are rotated in four directions then addded to the lists
              im_rotate = cv2.rotate(img, angle)           #this action is taken to increase the accuracy and expand the dataset
              #plt.imshow(im_rotate)
              #plt.show()
              x.append(im_rotate)
              y.append(label)
      except Exception as e:
          pass

import numpy as np
x = np.array(x)             #lists are transformed to numpy array
y = np.array(y)
x = x.astype(np.float32)    #to save space 
x = x/255                   #normalization
print(len(x))               #size of the dataset
from sklearn.model_selection import train_test_split    #train-test-validation sets are created
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10,shuffle=True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,shuffle=True)
print(len(x_train))
print(len(x_val))
print(len(x_test))

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation,BatchNormalization
import keras
from keras.optimizers import Adam

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model_checkpoint = ModelCheckpoint('/content/gdrive/My Drive/models/original.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
batch_size = 32
num_classes = 4
epochs = 100 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(patch_size,patch_size,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Adam(lr=0.00001),metrics=['accuracy'])

import tensorflow as tf
dot_img_file = '/content/gdrive/My Drive/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

hist = model.fit(x_train,y_train_one_hot, epochs=epochs, batch_size= batch_size, validation_data = (x_val,y_val_one_hot), callbacks=[early_stopping_monitor, model_checkpoint] )

model.summary()

print(model.metrics_names)
print(model.evaluate(x_test, y_test_one_hot))

from sklearn import metrics
y_pred_one_hot = model.predict(x_test)
y_pred_labels = np.argmax(y_pred_one_hot, axis = 1)
y_true_labels = np.argmax(y_test_one_hot,axis=1)
confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
print(confusion_matrix)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print('Accuracy: %f' % accuracy)

precision = precision_score(y_true_labels, y_pred_labels,pos_label='positive', average='micro')
print('Precision: %f' % precision)

recall = recall_score(y_true_labels, y_pred_labels, average='micro')
print('Recall: %f' % recall)

f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print('F1 score: %f' % f1)

from sklearn.metrics import classification_report
target_names = Categories
print(classification_report(y_true_labels, y_pred_labels, target_names=target_names))
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test, y_pred_labels)
print('Cohens kappa: %f' % kappa)

from sklearn import metrics
model = load_model('/content/gdrive/My Drive/models/original256.h5')