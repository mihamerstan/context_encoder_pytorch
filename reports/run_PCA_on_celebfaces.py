from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import pickle 

input_image_dir = '../nn_output/'
original_images_path_list = []
reconstructed_images_path_list = []

for file in os.listdir(input_image_dir):
    if file.endswith("orig.png"):
        original_images_path_list.append(file)
for file in os.listdir(input_image_dir):
    if file.endswith("recons.png"):
        reconstructed_images_path_list.append(file)
original_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in original_images_path_list]

all_pyramids_list = make_pyramids(original_images,3)
eig_vals_vecs_per_level = PCA_pyramids(all_pyramids_list)
    
for level, (eig_vals,eig_vecs) in enumerate(eig_vals_vecs_per_level):
    #eig_vals is sorted from lowest to highest
    #Take top 10 eigenvectors
    images = eig_vecs[-10:]
    images = np.flipud(images)
    plot_title = "CelebA Top 10 eigenvectors: " + level
    plot_image_grid(images,'Top 10 eigenvectors',image_shape=eig_vecs[0].shape)
pickle.dump(eig_vals_vecs_per_level, 'celebA_pyramids.p')
