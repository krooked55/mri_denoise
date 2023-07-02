#flipleft, flipupdown, zoom

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import random
from random import randint
import cv2 as cv

def image_space_slice(physio_im, quitar_slices):
    images = []
    n_append=0
    
    for j in physio_im:
        
        volume_kspace = j

        #list of slice in absolute value image space
        #Dimensión donde se encuentra el min
        shape=np.array(volume_kspace.shape)
        for k in list(range(volume_kspace.ndim)):
            if shape[k] == min(shape):
                dim_10 = k

        n_slice = min(shape)
        for i in list(range(quitar_slices, n_slice-quitar_slices)):

            #for m in list(range(min(shape))): 
            if dim_10 == 0:
                slice_kspace = volume_kspace[i,:,:]
            elif dim_10 == 1: 
                slice_kspace = volume_kspace[:,i,:]
            elif dim_10 == 2: 
                slice_kspace = volume_kspace[:,:,i]  
           
            #slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
            #slice_image = fastmri.ifft2c(slice_kspace2)

            #slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image
            #data = np.array(slice_image_abs)
            data = np.array(slice_kspace)
            data = abs(data)
            images.append(data)

    images=np.asarray(images)
    return images

def resize_and_crop(images, resize_factor):
    coord_center = [images.shape[0]/2, images.shape[1]/2]

    # Translate to resized coordinates
    h = images.shape[1]*resize_factor
    w = images.shape[0]*resize_factor

    cx, cy = [resize_factor*c for c in coord_center]

    im_array = images
    im_array = cv.resize(im_array, (0, 0), fx=resize_factor, fy=resize_factor)

    im_array = im_array[int(round(cy - h/resize_factor*0.5)) : int(round(cy + h/resize_factor*0.5)),
            int(round(cx - w/resize_factor*0.5)) : int(round(cx + w/resize_factor*0.5))]

    return im_array
    

def data_augmentation_methods(mats, direct, quitar_slices, SEED, x_train_shape0, x_train, x_train_noisy, flipleft, flipupdown, zoom):
    images = list(x_train)
    images = np.asarray(images)
    images_noise = list(x_train_noisy)
    images_noise = np.asarray(images_noise)

    #Flipleft (nScans=[1:2,:,:]) (realmente es más un flip transversal)
    if flipleft == 'si': 
        #Bucle flipleft GT flipleft
        images_flipleft = list(x_train)
        for i in list(range(x_train_shape0)):
            flipleft = np.flip(x_train[i,:,:])
            images_flipleft.append(flipleft)
        images=np.asarray(images_flipleft)
        print(images.shape)

        #Bucle flipleft noise
        #nScsans=2 flip left
        images_noise_flipleft=[]

        for i in mats:
            if 'checkpoints' in i:
                continue
            else:    
                naranja_mat = scipy.io.loadmat(direct + '/' + i)
                naranja = naranja_mat['imgFull']
                naranja_noise_flipleft = naranja[1:2,:,:,:]
                naranja_noise_flipleft = np.mean(naranja_noise_flipleft, axis=0)
                images_noise_flipleft.append(naranja_noise_flipleft)

        images_noise_flipleft = image_space_slice(images_noise_flipleft, quitar_slices)
        images_noise_flipleft, no_use = train_test_split(images_noise_flipleft,
                                        test_size=0.2,
                                        random_state=SEED)

        images_noise = list(x_train_noisy)
        for i in list(range(images_noise_flipleft.shape[0])):
            flipleft = np.flip(images_noise_flipleft[i,:,:])
            images_noise.append(flipleft)
        images_noise=np.asarray(images_noise)

        print(images_noise.shape)


    #Flipupdown (nScans=[2:3,:,:])
    if flipupdown == 'si': 
        #Bucle flipupdown GT
        images_flipupdown = list(images)
        for i in list(range(x_train_shape0)):
            flipupdown = np.flipud(images[i,:,:])
            images_flipupdown.append(flipupdown)
        images=np.asarray(images_flipupdown)
        print(images.shape)

        #Bucle flipud noise
        #nScsans=3 flip left
        images_noise_flipupdown=[]

        for i in mats:
            if 'checkpoints' in i:
                continue
            else:    
                naranja_mat = scipy.io.loadmat(direct + '/' + i)
                naranja = naranja_mat['imgFull']
                naranja_noise_flipupdown = naranja[2:3,:,:,:]
                naranja_noise_flipupdown = np.mean(naranja_noise_flipupdown, axis=0)
                images_noise_flipupdown.append(naranja_noise_flipupdown)

        images_noise_flipupdown = image_space_slice(images_noise_flipupdown, quitar_slices)
        images_noise_flipupdown, no_use = train_test_split(images_noise_flipupdown,
                                        test_size=0.2,
                                        random_state=SEED)

        images_noise = list(images_noise)
        for i in list(range(images_noise_flipupdown.shape[0])):
            flipupdown = np.flipud(images_noise_flipupdown[i,:,:])
            images_noise.append(flipupdown)
        images_noise=np.asarray(images_noise)
        print(images_noise.shape)


    #Zoom
    if zoom == 'si':
        #Bucle zoom noise
        images_noise_zoom=[]
        for i in mats:
            if 'checkpoints' in i:
                continue
            else:    
                naranja_mat = scipy.io.loadmat(direct + '/' + i)
                naranja = naranja_mat['imgFull']
                naranja_noise_zoom = naranja[2:3,:,:,:]
                naranja_noise_zoom = np.mean(naranja_noise_zoom, axis=0)
                images_noise_zoom.append(naranja_noise_zoom)
        images_noise_zoom = image_space_slice(images_noise_zoom, quitar_slices)
        images_noise_zoom, no_use = train_test_split(images_noise_zoom,
                                        test_size=0.2,
                                        random_state=SEED)
        images_noise = list(images_noise)
        images_zoom = list(images)
        for i in list(range(images_noise_zoom.shape[0])):
            resize_factor_range = [1.1, 1.7]
            resize_factor = round(random.uniform(resize_factor_range[0], resize_factor_range[1]), 2)
            #Bucle zoom noise
            zoom = resize_and_crop(images_noise_zoom[i,:,:], resize_factor)
            images_noise.append(zoom)
            #Bucle flipupdown GT
            zoom = resize_and_crop(images[i,:,:], resize_factor)
            images_zoom.append(zoom)            
        images_noise=np.asarray(images_noise)
        print(images_noise.shape)
    
        images=np.asarray(images_zoom)
        print(images.shape)  
    return images, images_noise