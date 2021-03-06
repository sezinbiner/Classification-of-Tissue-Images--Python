# -*- coding: utf-8 -*-
#This part is for google colab connection and image transfering
from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir("/content")
!unzip "/content/gdrive/My Drive/dataset/InSitu.zip"
!unzip "/content/gdrive/My Drive/dataset/benign.zip"
!unzip "/content/gdrive/My Drive/dataset/invasive.zip"

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

"""
In this function we divide the original images into 512*512 sized patches and eliminate the blank or useless patches with these steps
    - segment the original sized image by k-means algorithm
    - K-means gives color centers. Paint the lighter centers white so the black parts will be the cell cores
    - so the original image is converted to black and white. 
    - divide the black-white image and calculate the ratio of black area
    - it it is greater than the limit ratio take that part of th original image and add it to the dataset
"""

def k_means(img):
    """
    This functions is used for image segmentation.
    """
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4                                   #number of color centers
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    #print("center",center)
    colors, index = np.unique(center, return_index=True, return_inverse=False,return_counts=False, axis=0)
    #print("index",index,"colors",colors)
    center[index[0]] = (0,0,0)              #paint the lighter centers white and the darker centers black
    center[index[1]] = (0,0,0)
    center[index[2]] = (255,255,255)
    center[index[3]] = (255,255,255)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image                     #black-white image

SIZE1=512
angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

def isValid(img,category):
    """
    This function calculates the black area ratio and returns true if the patch is useful.
    """
    
    width,height = img.shape[:2]
    n_black_pix = np.sum(img == (0,0,0))
    n_black_pix /= 3
    ratio = (( n_black_pix ) / (width*height))*100
    #print("ratio",ratio)
    
    if (category != 'normal'):
        limit = 30
    else:
        limit = 10
        
    if (ratio > limit):
        return True,ratio
    else: 
        return False,ratio
        
def read_from_file(DATADIR,Categories):
    x = []                                                                                             #images
    y = []                                                                                             #labels
    for category in Categories: 
        path = os.path.join(DATADIR,category)
        os.chdir(path)
        filename = category
        k = 0
        for im in os.listdir(path): 
            
            if im is not None:
                img = cv2.imread(im,cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #plt.imshow(img)
                #plt.show()
                height,width = img.shape[:2]
                segmented_img = k_means(img)
                size = 512                                                                           #shift size of pixels  
                for i in range((height//SIZE1)):
                    for j in range(( width//size)):
                        segmented_crop = segmented_img[i*SIZE1:(i+1)*SIZE1, j*size:(j*size+SIZE1)]   #crop the black and white image
                        return_image,ratio = isValid(segmented_crop,category)
                        if return_image is not False:                                                #if the patch is useful crop the original image and save that patch
                            crop=img[i*SIZE1:(i+1)*SIZE1, j*size:(j*size+SIZE1)]
                            #plt.imshow(crop)
                            #plt.show()
                            filename = im
                            filename =filename + str(k) + ".tif"
                            print(filename)
                            k = k+1
                            path_save = os.path.join(DATADIR_SAVE,filename)
                            print(path_save)
                            cv2.imwrite(path_save, crop)
                img = cv2.resize(img,(SIZE1, SIZE1))
                filename = category
                filename =filename + str(k) + ".tif"                                        
                print(filename)
                k = k+1
                path_save = os.path.join(DATADIR_SAVE,filename)
                print(path_save)
                cv2.imwrite(path_save, img)                                                             #as this process takes too much time 
                                                                                                        #patches are saved to another file 
                
                for angle in angles:                                                                    #also add the original sized images and their rotations
                    im_rotate = cv2.rotate(img, angle)
                    filename = category
                    filename =filename + str(k) + ".tif"
                    print(filename)
                    k = k+1
                    path_save = os.path.join(DATADIR_SAVE,filename)
                    print(path_save)
                    cv2.imwrite(path_save, im_rotate)
                
            else:
              pass
    return x,y

DATADIR = "/content/gdrive/My Drive/2015/Test_data"                                                    #enter the path to read the images from
DATADIR_SAVE = "/content/gdrive/My Drive/Test_2015/Normal"                                             #enter the path to save the images to
Categories = ["Normal"]    
