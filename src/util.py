#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

SRC_DIR = 'images'
DEST_DIR = 'results'
#IMG_NAMES = ("desk.png","night_sample.jpg")
#IMG_NAMES = ("desk.png",)
IMG_NAMES = ("night_sample.jpg",)

def get_filenames():
    """Return list of tuples for source and template destination
       filenames(absolute filepath)."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    src_path = os.path.join(parent_dir, SRC_DIR)
    dest_path = os.path.join(parent_dir, DEST_DIR)
    filenames = []
    for name in IMG_NAMES:
        base, ext = os.path.splitext(name)
        tempname = base + '-%s' + ext
        filenames.append((os.path.join(src_path, name),
                          os.path.join(dest_path, tempname)))
    return filenames
