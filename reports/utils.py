from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from skimage.color import rgb2gray
import pickle

def plot_image_grid(images, title, image_shape=(64,64),n_col=5, n_row=2, bycol=0, row_titles=None,col_titles=None):
    fig,axes = plt.subplots(nrows=n_row,ncols=n_col,figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images):
        row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col)       
        cax = axes[row,col]
        cax.imshow(comp.reshape(image_shape), cmap='gray',
                   interpolation='nearest',
                   vmin=comp.min(), vmax=comp.max())
        cax.set_xticks(())
        cax.set_yticks(())
    if row_titles is not None :
        for ax,row in zip(axes[:,0],row_titles) :
            ax.set_ylabel(row,size='large')
    if col_titles is not None :
        for ax,col in zip(axes[0],col_titles) :
            ax.set_title(col)
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(title + '.pdf',bbox_inches='tight')
    plt.show()

def make_pyramids(images_list,num_levels):
    '''
    return struct: list_1 of list_2 of arrays
    list_2 has length equal to num_levels, and contains num_levels amount of pyramids
    list_1 has length equal to len(images_list)
    
    Access the return by all_pyramids_list[image_index][level_index]
    '''
    all_pyramids_list = []
    for image in images_list:
        prev_level = image
        one_pyramid_list = [image]
        for ii in range(num_levels):
            im_down = cv2.pyrDown(prev_level)
            prev_level = im_down
            one_pyramid_list.append(im_down)
        all_pyramids_list.append(one_pyramid_list)
    return all_pyramids_list

def PCA_pyramids(all_pyramids_list):
    '''
    for each level in each image pyramid, do a PCA on it, and store the principal components.
    
    !!ONLY HANDLES SQUARE IMAGES!!
    '''
    num_pyramids = len(all_pyramids_list)
    num_pixels = all_pyramids_list[0][0].shape[0]*all_pyramids_list[0][0].shape[1]
    num_levels = len(all_pyramids_list[0])
    eig_vals_vecs_per_level = []
    for level_idx in range(num_levels):
        num_pixels = all_pyramids_list[0][level_idx].shape[0]*all_pyramids_list[0][level_idx].shape[1]
        X = np.zeros((num_pyramids,num_pixels))

        for img_idx in range(num_pyramids):
            image = all_pyramids_list[img_idx][level_idx]
            #We "ravel" the image into a 1d array, we will have to unravel it later
            X[img_idx,:] = np.reshape(image,(-1,num_pixels))
        #Perform PCA on X
        cov = (X-X.mean(axis=0)).T@(X-X.mean(axis=0))
        eig_vals,eig_vecs = np.linalg.eigh(cov)
        #unravel each eigenvector
        new_eigvec_list = []
        for ii in range(len(eig_vecs)):
            num_pixels_per_side = int(num_pixels**.5)#this is square root of total pixels
            new_eig_vec = np.reshape(eig_vecs[:,ii],(num_pixels_per_side,num_pixels_per_side),'F')
            new_eigvec_list.append(new_eig_vec.T) # have to transpose for some reason? this isnt a good sign
        eig_vals_vecs_per_level.append((eig_vals,new_eigvec_list))
        
    return eig_vals_vecs_per_level


            
            