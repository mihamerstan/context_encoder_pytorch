from utils import plot_image_grid, make_pyramids, PCA_pyramids
import os
from skimage.color import rgb2gray
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
reconstructed_images = [rgb2gray(plt.imread(input_image_dir+file)) for file in reconstructed_images_path_list]

original_cropped = original_images[32:(32+64), 32:(32+64)]
reconstructed_cropped = reconstructed_cropped[32:(32+64), 32:(32+64)]


original_pyramids = make_pyramids(original_images,3)
orig_eig_per_level = PCA_pyramids(original_pyramids)
    
for level, (eig_vals,eig_vecs) in enumerate(orig_eig_per_level):
    #eig_vals is sorted from lowest to highest
    #Take top 10 eigenvectors
    images = eig_vecs[-10:]
    images = np.flipud(images)
    plot_title = "CelebA Top 10 eigenvectors: " + str(level)
    plot_image_grid(images,plot_title,image_shape=eig_vecs[0].shape)
pickle.dump(orig_eig_per_level, open('orig_celebA_pyramids.p','wb'))

del(original_pyramids)
del(orig_eig_per_level)

recon_pyramids = make_pyramids(reconstructed_images,3)
recon_eig_per_level = PCA_pyramids(recon_pyramids)
    
for level, (eig_vals,eig_vecs) in enumerate(recon_eig_per_level):
    #eig_vals is sorted from lowest to highest
    #Take top 10 eigenvectors
    images = eig_vecs[-10:]
    images = np.flipud(images)
    plot_title = "Reconstructed CelebA Top 10 eigenvectors: " + str(level)
    plot_image_grid(images,plot_title,image_shape=eig_vecs[0].shape)
pickle.dump(recon_eig_per_level, open('recon_celebA_pyramids.p','wb'))

del(recon_pyramids)
del(recon_eig_per_level)


cropped_original_pyramids = make_pyramids(original_cropped,3)
cropped_orig_eig_per_level = PCA_pyramids(cropped_original_pyramids)
    
for level, (eig_vals,eig_vecs) in enumerate(cropped_orig_eig_per_level):
    #eig_vals is sorted from lowest to highest
    #Take top 10 eigenvectors
    images = eig_vecs[-10:]
    images = np.flipud(images)
    plot_title = "Cropped_CelebA Top 10 eigenvectors: " + str(level)
    plot_image_grid(images,plot_title,image_shape=eig_vecs[0].shape)
pickle.dump(cropped_orig_eig_per_level, open('cropped_orig_celebA_pyramids.p','wb'))

del(cropped_original_pyramids)
del(cropped_orig_eig_per_level)

cropped_recon_pyramids = make_pyramids(reconstructed_cropped,3)
cropped_recon_eig_per_level = PCA_pyramids(cropped_recon_pyramids)
    
for level, (eig_vals,eig_vecs) in enumerate(cropped_recon_eig_per_level):
    #eig_vals is sorted from lowest to highest
    #Take top 10 eigenvectors
    images = eig_vecs[-10:]
    images = np.flipud(images)
    plot_title = "Cropped Reconstructed CelebA Top 10 eigenvectors: " + str(level)
    plot_image_grid(images,plot_title,image_shape=eig_vecs[0].shape)
pickle.dump(recon_eig_per_level, open('cropped_recon_celebA_pyramids.p','wb'))

del(cropped_recon_pyramids)
del(cropped_recon_eig_per_level)













