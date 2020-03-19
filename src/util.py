#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

SRC_DIR = 'images'
DEST_DIR = 'results'
#IMG_NAMES = ("desk.png","night_sample.jpg")
IMG_NAMES = ("from_MSR.png",)
#IMG_NAMES = ("cat.jpg",)
#IMG_NAMES = ("5101556393_resized.jpg",)


def get_filenames():
    """Return list of tuples for source and template destination
       filenames(absolute filepath)."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    src_path = os.path.join(parent_dir, SRC_DIR)
    dest_path = os.path.join(parent_dir, DEST_DIR)
    filenames = []
    global folder
    
    for name in IMG_NAMES:
        base, ext = os.path.splitext(name)
        tempname = base + '-%s' + ext
        filenames.append((os.path.join(src_path, name), os.path.join(dest_path, tempname)))
                          
        folder = filenames[0][0][:-4] + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)
                    
    return filenames
