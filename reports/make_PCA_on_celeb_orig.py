from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import pickle 

input_image_dir = '/scratch/awd275/context_encoder_pytorch_data/20200512_CE_output/'
output_file_path = '/scratch/awd275/context_encoder_pytorch_data/20200512_eigen_pyramids/celebA_1000_orig_eigen.p'
orig_images_path_list = []

for file in os.listdir(input_image_dir):
    if file.endswith("orig.png"):
        orig_images_path_list.append(file)

orig_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in orig_images_path_list]

orig_pyramids = make_pyramids(orig_images,3)
orig_eig_per_level = PCA_pyramids(orig_pyramids)

pickle.dump(orig_eig_per_level, open(output_file_path,'wb'))


