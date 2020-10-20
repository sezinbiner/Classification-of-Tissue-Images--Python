# Classification of Hematoxylin and Eosin Stained Tissue Images

In this project four types of cancerous tissues
- Benign
- In situ carcinoma
- Invasive carcinoma
- Normal <br> 
are classified using the convolutional neural network (CNN) which is a class of deep learning. Different models are tested and the images have been preprocessed. 
Python programming language is used. 
Working with tensorflow, keras, matplolib, scikitlearn, cv2, pillow, numpy libraries.
Images belong to Bach Iciar 2018 Challenge Dataset.
I highly recommend using Google Colab due to its computational power.

- Model 1 uses images original size resized to 256x256.
- Model 2 uses images divided into patched and the model is tested with size 256x256 and 512x512.
- In Transfer Learning the 256x256 sized patches are used and the model is trained with pre-trained models such as DenseNet, Inception, ResNet, VGG16, VGG19.
- use divide_images_into_patches.py to create patches and directories for them. Preprocessing happens in this file. <br>




