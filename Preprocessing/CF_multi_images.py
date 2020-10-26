
##################################################
## Content: This Script preprocesses images 
#  using kmeans colour space reduction

## Author: Stanislav Sys

## Last Update: 20.10.20

## Comments: This Script will automatically read
#  in all images in the directory where the script
#  is executed. Don't forget to Specifiy the image 
#  type as e.g ".jpg".
##################################################

# necessary imports

import os                                           # managing file
import numpy as np                                  # arrays for calcualtion
import matplotlib.pyplot as plt                     # create figures
from numpy.linalg import svd                        # perform SVD
from skimage import data                            # imageprocessing
import cv2											# image processing using openCV
from skimage.color import rgb2gray                  # -""-
import matplotlib.pyplot as plt                     # plotting
from PIL import Image                               # load images
import warnings; warnings.simplefilter('ignore')    # Fix NumPy issues.
from sklearn.cluster import MiniBatchKMeans         # K means algorithm
import concurrent.futures                           # multiprocessing 
import multiprocessing                              # multiprocessing classic
from datetime import datetime                       # get script running time


################### Constants

#enchmarking
startTime = datetime.now()


# path to images to be processed -> leave if this script should execute in CWD
dir_path = os.getcwd()+"/"

# number of colours the image will be downsampled to -> increasing hits on performance
k = 128

# number of singular Values to be used for image reconstruction after decomposition
#s = 50
kmeans = MiniBatchKMeans(k)


###################  Function definitions

# get image names
def get_image_names(directory_path):
    names = []
    filelist = os.listdir(directory_path)
    counter = 0

    for i in filelist:
        if i.endswith(".jpg"):
            counter = counter +1
            names.append(i)
             
    print("read in " + str(counter) + " files from : " + directory_path)
    return names
    
# downsample colours of images using kmeans algorithm and save them with matplotlib
def calculate_kmeans_figure_mpl(image_name):
    #first we have to load and reshape the image
    print(f'started job for {image_name}')
    image_np = np.array(Image.open(dir_path + image_name))
    image_2d = (image_np / 255)
    image_2d = image_2d.reshape(image_np.shape[0]*image_np.shape[1],image_np.shape[2])

    #now we use the kmeans algorithm to calculate new colours                                             
    kmeans.fit(image_2d)
    new_colors = kmeans.cluster_centers_[kmeans.predict(image_2d)]
    img_recolored = new_colors.reshape(image_np.shape)
    plt.figure(figsize =((image_np.shape[1]/1000), (image_np.shape[0]/1000)), dpi = 100)
    plt.imshow(img_recolored)
    plt.margins(0,0)
    plt.axis('off')
    plt.savefig(str(k) + "_kmeans_" + str(image_name), bbox_inches='tight', dpi=1000,pad_inches = 0)
    figure_name = "processed " + str(k) + "_kmeans_" + image_name 
    return figure_name

# downsample colours of images using kmeans algorithm and save them with cv2 -> better
def calculate_kmeans_figure_cv2(image_name):
    #first we have to load and reshape the image
    print(f'started job for {image_name}')
    image_np = np.array(Image.open(dir_path + image_name))
    image_2d = (image_np / 255)
    image_2d = image_2d.reshape(image_np.shape[0]*image_np.shape[1],image_np.shape[2])

    #now we use the kmeans algorithm to calculate new colours                                             
    kmeans.fit(image_2d)
    new_colors = kmeans.cluster_centers_[kmeans.predict(image_2d)]
    img_recolored = new_colors.reshape(image_np.shape)
    # cv2 only accepts integers and Color values in BGR -> convert	
    ima2 = (img_recolored[..., ::-1] * 255)
    cv2.imwrite("kmeans/kmeans_"+ str(k) + "_" +  str(image_name),ima2)
    figurename = "Downsampled {} to {} colours".format(image_name, k)
    return figurename

################### main script

# get filenames
name_list = get_image_names(dir_path)

# use concurrent futures for for multiprocessing -> adapt workers ( 4 were the seetspot on my machine)
with concurrent.futures.ProcessPoolExecutor(max_workers= 4) as executor:
    results = [executor.submit(calculate_kmeans_figure_cv2, filename) for filename in name_list]

    for f in concurrent.futures.as_completed(results):
        print(f.result())
# end benchmark
print(datetime.now() - startTime)
