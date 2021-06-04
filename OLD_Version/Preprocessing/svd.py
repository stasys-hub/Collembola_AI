##################################################
## Content: This Script preprocesses images 
#  using SVD 

## Author: Stanislav Sys

## Last Update: 20.10.20

## Comments: This Script will automatically read
#  in all images in the directory where the script
#  is executed. 
##################################################

# necessary imports
import numpy as np							# np arrays for calcalutation
from PIL import Image							# loading images
import matplotlib.pyplot as plt						# plotting
from numpy.linalg import svd						# numpys svd implementation
from skimage import data 						# image handling
from skimage.color import rgb2gray					# -||-						
import cv2                                                              # image processing using openCV
import os								# directory handling
import concurrent.futures                                               # multiprocessing 
import multiprocessing                                                  # multiprocessing classic
from datetime import datetime						# benchmarking



################### Constants
# time for benchmarking
startTime = datetime.now()

# get dir path
dir_path = os.getcwd()+"/"

# number of singular values used for image reconstruction
d = 300

# needed to reconstruct image in right pixel size -> get your display dpi here: https://www.infobyip.com/detectmonitordpi.php
my_dpi = 96

################### Function Definition

# get the names of images in your CWD
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

# calculate svd for an image
def compress_image_svd(image):
    # Define the matrices for decompostion and calculate the svd of the image:
    U,s,V = svd(image, full_matrices=False)
    leng = len(s)
    #print(f'image contains {leng singular values')
    # Reconstruct the matrix by calculating the dot product of the decomposition using only k singular values
    # note that we are using np.diag for the singular values, since they are in array form
    reconstimg = np.matrix(U[:, :d]) * np.diag(s[:d]) * np.matrix(V[:d, :])
    # return the reconstructed matrix and the array of singular values
    return reconstimg

# plot and save reconstrcuted image
def plot_svd_image_np(image_name):

    print(f"job for {image_name} started")
    image = np.array(Image.open(dir_path + image_name))
    reconstructed_layers = [compress_image_svd(image[:,:,i]) for i in range(3)]
    recon_image = np.zeros(image.shape)
    for j in range(3):
        recon_image[:,:,j] = reconstructed_layers[j]
    plt.figure(figsize =((recon_image.shape[1]/1000), (recon_image.shape[0]/1000)), dpi = my_dpi)
    plt.margins(0,0)
    plt.axis('off')
    plt.imshow(recon_image / 255.0)
    plt.savefig(str(d) + '_svd_' + image_name, bbox_inches='tight', pad_inches = 0, dpi = my_dpi * 10)
    return 'svd computed for: ', image_name

def plot_svd_image_cv2(image_name):

    print(f"job for {image_name} started")
    image = np.array(Image.open(dir_path + image_name))
    reconstructed_layers = [compress_image_svd(image[:,:,i]) for i in range(3)]
    recon_image = np.zeros(image.shape)
    for j in range(3):
        recon_image[:,:,j] = reconstructed_layers[j]
    # cv2 uses BGR, we have to adapt it to RGB
    ima2 = recon_image[..., ::-1]
    cv2.imwrite("svd/svd_"+ str(d) + "_" + str(image_name), ima2)
    figurename = "Reconstructed {} with {} singular values".format(image_name, d)
    return figurename
################### Main Script 

# get filenames
name_list = get_image_names(dir_path)

# use concurrent futures for for multiprocessing -> adapt workers ( 4 were the seetspot on my machine)
with concurrent.futures.ProcessPoolExecutor(max_workers= 4) as executor:
    results = [executor.submit(plot_svd_image_cv2, filename) for filename in name_list]

    for f in concurrent.futures.as_completed(results):
        print(f.result())
#end benchmark
print(datetime.now() - startTime)
