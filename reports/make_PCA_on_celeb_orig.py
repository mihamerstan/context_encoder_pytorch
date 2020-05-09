from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import pickle 

print('starting')
input_image_dir = '../nn_output/'
original_images_path_list = []

for file in os.listdir(input_image_dir):
    if file.endswith("orig.png"):
        original_images_path_list.append(file)
print('files read')
original_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in original_images_path_list]

original_pyramids = make_pyramids(original_images,3)
orig_eig_per_level = PCA_pyramids(original_pyramids)
print('PCA pyramids made')   

pickle.dump(orig_eig_per_level, open('celebA_orig_eigen.p','wb'))
print('dumped')
del(original_pyramids)
del(orig_eig_per_level)
print('ending')
