from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import pickle 


input_image_dir = '/scratch/awd275/context_encoder_pytorch_output/20200512_output/'
original_images_path_list = []

for file in os.listdir(input_image_dir):
    if file.endswith("orig.png"):
        original_images_path_list.append(file)

original_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in original_images_path_list]
original_cropped = [image[32:(32+64), 32:(32+64)] for image in original_images]
del(original_images)

cropped_original_pyramids = make_pyramids(original_cropped,3)
cropped_orig_eig_per_level = PCA_pyramids(cropped_original_pyramids)
    
pickle.dump(cropped_orig_eig_per_level, open('celebA_1000_orig_cropped_eigen.p','wb'))

del(cropped_original_pyramids)
del(cropped_orig_eig_per_level)
