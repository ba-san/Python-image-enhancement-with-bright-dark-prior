#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from guidedfilter import guided_filter
from PIL import Image
import util

L = 256  # color depth

def get_illumination_channel(I, w): # (1) No problem in this func. 100% sure.
    M, N, _ = I.shape
    padded = np.pad(I, ((w/2, w/2), (w/2, w/2), (0, 0)), 'edge') # edge can be changed. Run 'python2' for this line.

    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))
    
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j]  =  np.min(padded[i:i + w, j:j + w, :])
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])
    
    return darkch, brightch
    
def get_atmosphere(I, brightch, p): # (2) Theoretically, there seems to be no mistakes.
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)
    flatbright = brightch.ravel() # make array flatten
    
    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # find top M * N * p indexes. argsort() returns sorted (ascending) index.
    
    # return the mean intensity for each channel
    A = np.mean(flatI.take(searchidx, axis=0),dtype=np.float64, axis=0) # take get value from index.
    
    return A

def get_initial_transmission(A, brightch): # (3) There seems to be little problem for now?
    A_c = np.max(A)
    
    init_t = (brightch-A_c)/(1.-A_c) # also referred to He's (3)
    init_t = (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # min-max normalization.
    
    return init_t # also referred to He's (3)

    
def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, w): # (4)?
    im3 = np.empty(I.shape, I.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = I[:,:,ind]/A[ind]

    dark_c, _ = get_illumination_channel(im3, w)
    dark_t = 1 - 0.7*dark_c # change here not to make minus number?
    #print('dark channel prior between [%.4f, %.4f]' % (dark_c.min(), dark_c.max())) #dark channel prior between [0.0000, 1.5222]
    #print('initial dark transmission rate between [%.4f, %.4f]' % (dark_t.min(), dark_t.max()))

    #######################################################
    white = np.full_like(dark_c, L - 1) # this is Idark shape array which all elements are 255.
    cv2.imwrite(util.folder + '/dark_t.png', dark_t*white)
    print('initial dark transmission rate between [%.4f, %.4f]' % (dark_t.min(), dark_t.max()))

    J_dark = get_final_image(I, A, dark_t, 0.1) # no problem for this func.
    cv2.imwrite(util.folder + '/J_dark.png', J_dark*255)
    #######################################################
    
    corrected_t = init_t
    diffch = brightch - darkch
    
    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i,j]<alpha): # direction of inequality sign doesn't matter.
                corrected_t[i,j] = dark_t[i,j]*init_t[i,j] # (13)

    return np.abs(corrected_t), dark_t
    

def get_final_image(I, A, corrected_t, tmin): # (6) No problem for this func. This is compatible with 'Recover' func.
    corrected_t_broadcasted = np.broadcast_to(corrected_t[:,:,None], (corrected_t.shape[0], corrected_t.shape[1], 3))
    J = (I-A)/(np.where(corrected_t_broadcasted < tmin, tmin, corrected_t_broadcasted)) + A
    return J
    

def dehaze_raw(I, tmin, w, p, eps):
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w) # (1) no problem for this func.
    print("dark max:{} min:{}".format(np.max(Idark), np.min(Idark))) # dark max:0.776470588235 min:0.0
    print("bright max:{} min:{}".format(np.max(Ibright), np.min(Ibright))) # bright max:0.988235294118 min:0.00392156862745

    cv2.imwrite(util.folder + '/init_bright_channel_prior.png', Ibright*255)
    cv2.imwrite(util.folder + '/init_dark_channel_prior.png', Idark*255)
    
    white = np.full_like(Idark, L - 1) # this is Idark shape array which all elements are 255.
    
    A = get_atmosphere(I, Ibright, p) # (2)
    print('atmosphere', A)

    #################################################################
    
    init_t = get_initial_transmission(A, Ibright) # (3)
    cv2.imwrite(util.folder + '/initial.png', init_t*white)
    print('initial transmission rate between [%.4f, %.4f]' % (init_t.min(), init_t.max()))

    J_init = get_final_image(I, A, init_t, tmin) # no problem for this func.
    cv2.imwrite(util.folder + '/J_init.png', J_init*255)
    after_init_I = I
    
    #################################################################
    
    corrected_t, dark_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, 0.1, w) # (4)
    cv2.imwrite(util.folder + '/corrected.png', corrected_t*white)
    print('corrected transmission rate between [%.4f, %.4f]' % (corrected_t.min(), corrected_t.max()))

    J_corrected = get_final_image(I, A, corrected_t, tmin)
    cv2.imwrite(util.folder + '/J_corrected.png', J_corrected*255)

    #################################################################
    
    # guided filter (5)?
    normI = (I - I.min()) / (I.max() - I.min())  # min-max normalize I
    refined_t = guided_filter(normI, corrected_t, w, eps)
    refined_dark_t = guided_filter(normI, dark_t, w, eps)
    
    cv2.imwrite(util.folder + '/refined.png', refined_t*white)
    print('refined transmission rate between [%.4f, %.4f]' % (refined_t.min(), refined_t.max()))

    J_refined = get_final_image(I, A, refined_t, tmin)
    cv2.imwrite(util.folder + '/J_refined.png', J_refined*255)

    cv2.imwrite(util.folder + '/refined_dark.png', refined_dark_t*white)
    print('refined dark transmission rate between [%.4f, %.4f]' % (refined_dark_t.min(), refined_dark_t.max()))

    J_refined_dark = get_final_image(I, A, refined_dark_t, tmin)
    cv2.imwrite(util.folder + '/J_refined_dark.png', J_refined_dark*255)
        
    return Idark, Ibright, A, init_t, refined_t


def dehaze(im, tmin=0.1, w=15, p=0.1, eps=1e-3):
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
    cv2.imwrite(util.folder + '/J_original.png', im)
    I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
    I = I[:,:,:3]/255 # See here. https://stackoverflow.com/questions/44955656/how-to-convert-rgb-pil-image-to-numpy-array-with-3-channels
    Idark, Ibright, A, init_t, refined_t = dehaze_raw(I, tmin, w, p, eps)
    white = np.full_like(Idark, L - 1)

    #def to_img(raw):
        #cut = np.maximum(np.minimum(raw, L - 1), 0).astype(np.uint8) # threshold to [0, L-1]

        #if len(raw.shape) == 3:
            #print('Range for each channel:')
            #for ch in xrange(3):
                #print('[%.2f, %.2f]' % (raw[:, :, ch].max(), raw[:, :, ch].min()))
            #return Image.fromarray(cut)
        #else:
            #return Image.fromarray(cut)

    #return [to_img(raw) for raw in (Idark, Ibright, white * init_t, white * refined_t, get_final_image(I, A, init_t, tmin), get_final_image(I, A, refined_t, tmin))]
