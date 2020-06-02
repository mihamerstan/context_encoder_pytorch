from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
#import cv2 #Moved into make_pyramids
import os
from skimage.color import rgb2gray
import pickle

def plot_image_grid(images, 
                    title, 
                    image_shape=(64,64),
                    ncols=5,
                    nrows=2, 
                    bycol=0, 
                    row_titles=None,
                    col_titles=None,
                    save=False):
    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(2. * n_col, 2.26 * n_row))
    for i, image in enumerate(images):
        row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col) 
        if nrow==1:
            cax = axes[col]
        else:
            cax = axes[row,col]
        cax.imshow(image.reshape(image_shape), cmap='gray',
                   interpolation='nearest',
                   vmin=image.min(), vmax=image.max())
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
    if save is True:
        plt.savefig(title + '.pdf',bbox_inches='tight')
    plt.show()

def make_pyramids(images_list,num_levels):
    '''
    return struct: list_1 of list_2 of arrays
    list_2 has length equal to num_levels, and contains num_levels amount of pyramids
    list_1 has length equal to len(images_list)
    
    Access the return by all_pyramids_list[image_index][level_index]
    '''
    import cv2 # did this here because jupyter has trouble finding environments, and cv2 isn't installed by default...
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
            #We "unravel" the image into a 1d array, we will have to "ravel" it later
            X[img_idx,:] = np.reshape(image,(-1,num_pixels))
        #Perform PCA on X
        cov = (X-X.mean(axis=0)).T@(X-X.mean(axis=0))
        eig_vals,eig_vecs = np.linalg.eigh(cov)
        #"ravel" each eigenvector
        new_eigvec_list = []
        for ii in range(len(eig_vecs)):
            num_pixels_per_side = int(num_pixels**.5)#this is square root of total pixels
            new_eig_vec = ravel_image_vec(eig_vecs[:,ii],num_pixels_per_side)
            
            new_eigvec_list.append(new_eig_vec) 
        eig_vals_vecs_per_level.append((eig_vals,new_eigvec_list))
        
    return eig_vals_vecs_per_level

def unravel_image(image):
    num_pixels = image.shape[0]*image.shape[1]
    image_vector = np.reshape(image,(-1,num_pixels))
    return image_vector
def ravel_image_vec(image_vector,num_pixels_per_side):
    image = np.reshape(image_vector,(num_pixels_per_side,num_pixels_per_side),'F')
    image = image.T
    return image


    
                

            
            