# -*- coding: utf-8 -*-

"""
This project was compiled in Google Colab and ı advise you to do the same because of its computational power.
"""

#To transfer datasets from drive 
from google.colab import drive
drive.mount('/content/gdrive')
!unzip "/content/gdrive/My Drive/copy/benign.zip"
!unzip "/content/gdrive/My Drive/copy/insitu.zip"
!unzip "/content/gdrive/My Drive/copy/invasive.zip"
!unzip "/content/gdrive/My Drive/copy/normal.zip"

DATADIR = "/content" 
Categories = ["benign","insitu","invasive","normal"] 

import os
import cv2
x = []
y = []
patch_size = 224                                            #accepted size by the trained model
for category in Categories:
  path = os.path.join(DATADIR,category)
  print(path)
  os.chdir(path) 
  label = Categories.index(category)                        #label 0->Benign 1-> InSitu 2->Invasive 3->Normal
  for im in os.listdir(path):
      try:
          img = cv2.imread(im,cv2.IMREAD_COLOR)             #reading patches cv2 reads images in BGR format
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #transform images into RGB format
          img = cv2.resize(img,(patch_size, patch_size))    #resize all images to same size 256*256
          """
          import matplotlib.pyplot as plt
          plt.imshow(img)
          plt.show()
          """
          x.append(img)                                  #in x images are stored
          y.append(label)                                #in y labels are stored
      except Exception as e:                             
          pass

import numpy as np
x = np.array(x)                                         #lists are transformed to numpy array
y = np.array(y)
x = x.astype(np.float32)                                #to save space 
x = x/255                                               #normalization
print(f'total number of images: {len(x)}')              #size of the dataset


print(f'total number of images class {Categories[0]} : {np.sum(y == 0)}')
print(f'total number of images class {Categories[1]} : {np.sum(y == 1)}')
print(f'total number of images class {Categories[2]} : {np.sum(y == 2)}')
print(f'total number of images class {Categories[3]} : {np.sum(y == 3)}')

#dataset is unbalanced to equalize the number of each class' 
#samples use 'Random Under Sampler'
#used before splitting the dataset in order to keep the sample size high

from imblearn.under_sampling import RandomUnderSampler 
random_under_sampler = RandomUnderSampler('auto')
x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2] * x.shape[3]))
x_rus, y_rus = random_under_sampler.fit_sample(x, y)
x_rus = x_rus.reshape(x_rus.shape[0], patch_size, patch_size, 3)

print(f'total number of images class {Categories[0]} : {np.sum(y_rus == 0)}')
print(f'total number of images class {Categories[1]} : {np.sum(y_rus == 1)}')
print(f'total number of images class {Categories[2]} : {np.sum(y_rus == 2)}')
print(f'total number of images class {Categories[3]} : {np.sum(y_rus == 3)}')

from sklearn.model_selection import train_test_split     #train-test-validation sets are created
x_train, x_test, y_train, y_test = train_test_split(x_rus,y_rus,test_size=0.10,shuffle=True)

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,shuffle=True)
print(f'total number of training images: {len(x_train)}')
print(f'total number of validation images: {len(x_val)}')
print(f'total number of test images: {len(x_test)}')

from keras.utils import to_categorical                   #one-hot-encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
y_val_one_hot = to_categorical(y_val)

BATCH_SIZE = 16                                          #to prevent the session from crashing keep the size as small as you can
from keras.preprocessing.image import ImageDataGenerator
train_generator = ImageDataGenerator(zoom_range=2, rotation_range = 90, horizontal_flip=True, vertical_flip=True)

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import RMSprop

def build_model(base_model, lr=1e-4):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())                 #this also flattens the data
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model

from keras.applications import DenseNet201, InceptionV3, VGG16, VGG19, ResNet50         #pre-trained models

print("Choose a model to train \n 1: DenseNet201 \n 2:InceptionV3 \n 3:VGG19 \n 4:VGG16 \n 5:ResNet50")
choice = input()

if choice == '1':  
  densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224,224,3))
  model = build_model(densenet ,lr = 1e-4)
elif choice == '2':
  inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
  model = build_model(inception ,lr = 1e-4)
elif choice == '3':
  vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
  model = build_model(vgg19 ,lr = 1e-4)
elif choice == '4':
  vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  model = build_model(vgg16 ,lr = 1e-4)
else:
  resnet = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))
  model = build_model(resnet ,lr = 1e-4)

model.summary()

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1,factor=0.2, min_lr=1e-7)
model_checkpoint = ModelCheckpoint('/content/gdrive/My Drive/git_hub_models/densenet201.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit_generator(
    train_generator.flow(x_train, y_train_one_hot, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=50,
    validation_data=(x_val, y_val_one_hot),
    callbacks=[learn_control, model_checkpoint]
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
