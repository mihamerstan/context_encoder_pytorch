from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import pickle 


input_image_dir = '/scratch/awd275/context_encoder_pytorch_data/20200512_CE_output/'
output_file_path = '/scratch/awd275/context_encoder_pytorch_data/20200512_eigen_pyramids/celebA_1000_recons_eigen.p'
reconstructed_images_path_list = []

for file in os.listdir(input_image_dir):
    if file.endswith("recons.png"):
        reconstructed_images_path_list.append(file)

reconstructed_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in reconstructed_images_path_list]

recon_pyramids = make_pyramids(reconstructed_images,3)
recon_eig_per_level = PCA_pyramids(recon_pyramids)

pickle.dump(recon_eig_per_level, open(output_file_path,'wb'))

