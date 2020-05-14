from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import pickle 


input_image_dir = '/scratch/awd275/context_encoder_pytorch_data/20200512_eigen_pyramids/20200512_output/'
output_file_path = '/scratch/awd275/context_encoder_pytorch_data/20200512_eigen_pyramids/celebA_1000_orig_cropped_eigen.p'
reconstructed_images_path_list = []

for file in os.listdir(input_image_dir):
    if file.endswith("recons.png"):
        reconstructed_images_path_list.append(file)

reconstructed_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in reconstructed_images_path_list]
reconstructed_cropped = [image[32:(32+64), 32:(32+64)] for image in reconstructed_images]
del(reconstructed_images)

cropped_recon_pyramids = make_pyramids(reconstructed_cropped,3)
cropped_recon_eig_per_level = PCA_pyramids(cropped_recon_pyramids)

pickle.dump(cropped_recon_eig_per_level, open(output_file_path,'wb'))

del(cropped_recon_pyramids)
del(cropped_recon_eig_per_level)

