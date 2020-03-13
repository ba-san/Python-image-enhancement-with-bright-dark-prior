#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from PIL import Image
from guidedfilter import guided_filter

R, G, B = 0, 1, 2  # index for convenience
L = 256  # color depth

def get_illumination_channel(I, w): # (1) No problem in this func.
    """Get the dark channel prior in the (RGB) image data.

    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size

    Return
    -----------
    An M * N array for the dark/bright channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((w/2, w/2), (w/2, w/2), (0, 0)), 'edge') # edge can be changed.

    # Don't do this. --> darkch = brightch = np.zeros((M, N))
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j]  =  np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])

    return darkch, brightch

def get_atmosphere(I, brightch, p): # (2)
    """Get the atmosphere light in the (RGB) image data.

    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    brightch: the bright channel prior of the image as an M * N numpy array
    p:      percentage of pixels for estimating the atmosphere light

    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    #test = [0,1,2,3,4,5,6,7,8,9]
    #print(test[:3]) # [0, 1, 2]
    
    # reference CVPR09, 4.4
    M, N = brightch.shape
    flatI = I.reshape(M * N, 3)
    flatbright = brightch.ravel() # make array flatten
    searchidx = (-flatbright).argsort()[:int(M * N * p)]  # find top M * N * p indexes. argsort() returns sorted(ascending) index.
    #searchidx = (flatbright).argsort()[:int(M * N * p)]  # find top M * N * p indexes. argsort() returns sorted(ascending) index.
    #print('for atmosphere')
    #print(flatI.take(searchidx, axis=0)[:10])
    #exit()
    
    # return the mean intensity for each channel
    return np.mean(flatI.take(searchidx, axis=0),dtype=np.int64, axis=0) # take get value from index.

def get_initial_transmission(A, brightch): # (3)?
    """Get the initial transmission esitmate in the (RGB) image data.

    Parameters
    -----------
    A:         a 3-element array containing atmosphere light ([0, L-1]) for each channel
    brightch:  the bright channel prior of the image as an M * N numpy array

    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    A_c = np.max(A)
    init_t = (brightch-A_c)/(255.-A_c) # also referred to He's (3)

    #init_t = np.where((brightch-A_c)/(255.-A_c) < 0., 0., (brightch-A_c)/(255.-A_c))  # threshold
    init_t = (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # min-max normalization. this is best for now.
    
    return init_t # also referred to He's (3)

    
def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, w): # (4)?
    """Get the transmission esitmate in the (RGB) image data.

    Parameters
    -----------
    I:        the M * N * 3 RGB image data ([0, L-1]) as numpy array
    A:        a 3-element array containing atmosphere light ([0, L-1]) for each channel
    darkch:   the dark channel prior of the image as an M * N numpy array
    brightch: the bright channel prior of the image as an M * N numpy array
    init_t:   bias for the estimate
    alpha:    the threshold for diffch
    w:        window size for the estimate

    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    dark_c, _ = get_illumination_channel(I/A, w)
    dark_t = 1 - dark_c
    corrected_t = init_t
    diffch = brightch - darkch

    cnt = 0
    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i,j]<alpha):
                cnt = cnt + 1
                corrected_t[i,j] = dark_t[i,j]*init_t[i,j] # (13)

    print(cnt)
    return np.abs(corrected_t)
    

def get_final_image(I, A, corrected_t, tmin): # (6) No problem for this func.
    corrected_t_broadcasted = np.broadcast_to(corrected_t[:,:,None], (corrected_t.shape[0], corrected_t.shape[1], 3))
    J = (I-A)/(np.where(corrected_t_broadcasted < tmin, tmin, corrected_t_broadcasted)) + A
    return J


def DarkChannel(im,sz): # range is 0.0 to 1.0
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b); # smallest in 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

    
def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[ind]
    #exit()

    transmission = 1 - omega*DarkChannel(im3,sz);
    #print(transmission)
    #print(transmission.shape)
    return transmission
    

def dehaze_raw(I, tmin=0.1, w=15, p=0.1, r=40, eps=1e-3):
    """Get the dark channel prior, atmosphere light, transmission rate
       and refined transmission rate for raw RGB image data.

    Parameters
    -----------
    I:      M * N * 3 data as numpy array for the hazy image
    tmin:   threshold of transmission rate
    w:      window size of the dark channel prior
    p:      percentage of pixels for estimating the atmosphere light
    r:      the radius of the guidance
    eps:    epsilon for the guided filter

    Return
    -----------
    (Idark, A, rawt, refined_t) if guided=False, then rawt == refined_t
    """
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w) # (1)

    white = np.full_like(Idark, L - 1) # this is Idark shape array which all elements are 255.

    A = get_atmosphere(I, Ibright, p) # (2)
    print('atmosphere', A)

    init_t = get_initial_transmission(A, Ibright) # (3)
    cv2.imwrite('initial.png', init_t*white)
    cv2.imwrite('init_bright.png', Ibright)
    print('initial transmission rate between [%.4f, %.4f]' % (init_t.min(), init_t.max()))

    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, 10, w) # (4)
    cv2.imwrite('corrected.png', corrected_t*white)
    print('corrected transmission rate between [%.4f, %.4f]' % (corrected_t.min(), corrected_t.max()))
    
    corrected_t = np.maximum(corrected_t, tmin)  # threshold t
    refined_t = np.maximum(corrected_t, tmin)  # threshold t
    
    # guided filter (5)?
    normI = (I - I.min()) / (I.max() - I.min())  # min-max normalize I
    refined_t = guided_filter(normI, refined_t, r, eps)
    
    cv2.imwrite('refined.png', refined_t*white)
    print('refined transmission rate between [%.4f, %.4f]' % (refined_t.min(), refined_t.max()))

    J_corrected = get_final_image(I, A, corrected_t, tmin) # (6)
    cv2.imwrite('J_corrected.png', J_corrected)

    J_refined = get_final_image(I, A, refined_t, tmin)
    cv2.imwrite('J_refined.png', J_refined)

    exit()
        
    return Idark, Ibright, A, init_t, refined_t


def dehaze(im, tmin=0.1, w=15, p=0.1, r=40, eps=1e-3):
    """Dehaze the given RGB image.

    Parameters
    ----------
    im:     the Image object of the RGB image
    other parameters are the same as `dehaze_raw`

    Return
    ----------
    (dark, init_t, refined_t, rawrad, rerad)
    Images for dark channel prior, raw transmission estimate,
    refiend transmission estimate, recovered radiance with raw t,
    recovered radiance with refined t.
    """
    I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
    #I = I/255; # 0 to 1
    I = I[:,:,:3] # See here. https://stackoverflow.com/questions/44955656/how-to-convert-rgb-pil-image-to-numpy-array-with-3-channels
    Idark, Ibright, A, init_t, refined_t = dehaze_raw(I, tmin, w, p, r, eps)
    white = np.full_like(Idark, L - 1)

    def to_img(raw):
        cut = np.maximum(np.minimum(raw, L - 1), 0).astype(np.uint8) # threshold to [0, L-1]

        if len(raw.shape) == 3:
            print('Range for each channel:')
            for ch in xrange(3):
                print('[%.2f, %.2f]' % (raw[:, :, ch].max(), raw[:, :, ch].min()))
            return Image.fromarray(cut)
        else:
            return Image.fromarray(cut)

    return [to_img(raw) for raw in (Idark, Ibright, white * init_t, white * refined_t, get_final_image(I, A, init_t, tmin), get_final_image(I, A, refined_t, tmin))]
