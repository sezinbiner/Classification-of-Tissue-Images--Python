# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/gdrive')

""""
Patches sized 256*256

Benign : 1789 
İnsitu : 1964 
İnvasive : 2205 
Normal : 1557 
Total: 7515 
""""

import os
import cv2
DATADIR = "/content/gdrive/My Drive/256" 
Categories = ["Benign","Insitu","Invasive","Normal"] 
angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

x = []
y = []
patch_size = 256
for category in Categories:
  path = os.path.join(DATADIR,category)
  print(path)
  os.chdir(path) 
  label = Categories.index(category)  #etiket
  for im in os.listdir(path):
      try:
          img = cv2.imread(im,cv2.IMREAD_COLOR)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = cv2.resize(img,(patch_size, patch_size))
          """
          import matplotlib.pyplot as plt
          plt.imshow(img)
          plt.show()
          """
          x.append(img)
          y.append(label)
      except Exception as e:
          pass

import numpy as np
x = np.array(x)
y = np.array(y)
x = x.astype(np.float32)  
x = x/255
print(f'Total number of images: {len(x)}')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10,shuffle=True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,shuffle=True)
print(f'total number of training images: {len(x_train)}')
print(f'total number of validation images: {len(x_val)}')
print(f'total number of test images: {len(x_test)}')

from imblearn.under_sampling import RandomUnderSampler
random_under_sampler = RandomUnderSampler('auto')

print(f'total number of images class {Categories[0]} : {np.sum(y_train == 0)}')
print(f'total number of images class {Categories[1]} : {np.sum(y_train == 1)}')
print(f'total number of images class {Categories[2]} : {np.sum(y_train == 2)}')
print(f'total number of images class {Categories[1]} : {np.sum(y_train == 3)}')


x_train = x_train.reshape(x_train.shape[0], (x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
x_train_rus,y_train_rus = random_under_sampler.fit_sample(x_train, y_train)
x_train = x_train_rus.reshape(x_train_rus.shape[0], patch_size, patch_size, 3)

print(f'total number of images class {Categories[0]} : {np.sum(y_train_rus == 0)}')
print(f'total number of images class {Categories[1]} : {np.sum(y_train_rus == 1)}')
print(f'total number of images class {Categories[2]} : {np.sum(y_train_rus == 2)}')
print(f'total number of images class {Categories[1]} : {np.sum(y_train_rus == 3)}')

from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_rus)
y_test_one_hot = to_categorical(y_test)
y_val_one_hot = to_categorical(y_val)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation,BatchNormalization
import keras
from keras.optimizers import Adam,RMSprop

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=25, mode='min')
model_checkpoint = ModelCheckpoint('/content/gdrive/My Drive/models/original256.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
batch_size = 32
patch_size = 256
num_classes = 4
epochs = 250
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=((patch_size,patch_size,3)), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))    

model.compile(loss='categorical_crossentropy',
          optimizer=RMSprop(lr=1e-4),
          metrics=['accuracy'])
model.summary()

BATCH_SIZE = 16
from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(
        zoom_range=2,  
        rotation_range = 90,
        horizontal_flip=True,  
        vertical_flip=True,  
    )

history = model.fit_generator(
    train_generator.flow(x_train, y_train_one_hot, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=200,
    validation_data=(x_val, y_val_one_hot),
    callbacks=[model_checkpoint]
)

print(model.metrics_names)
print(model.evaluate(x_test, y_test_one_hot))
from sklearn import metrics
y_pred_one_hot = model.predict(x_test)
y_pred_labels = np.argmax(y_pred_one_hot, axis = 1)
y_true_labels = np.argmax(y_test_one_hot,axis=1)
confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
print(confusion_matrix)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_true_labels, y_pred_labels,pos_label='positive', average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_true_labels, y_pred_labels, average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print('F1 score: %f' % f1)
from sklearn.metrics import classification_report
target_names = Categories
print(classification_report(y_true_labels, y_pred_labels, target_names=target_names))
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test, y_pred_labels)
print('Cohens kappa: %f' % kappa)
