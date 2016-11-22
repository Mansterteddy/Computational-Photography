import os 
import numpy as np 

from PIL import Image

SRC_DIR = 'img'
DEST_DIR = 'result'
IMG_NAMES = ("canon3.bmp", "gugong.bmp", "hongkong.bmp", "ny1.bmp", "tiananmen1.bmp", "train.bmp")

def get_filenames():
    """
    Return list of tuples for source and template destination filenames(absolute filepath).
    """
    #get current file path
    file_dir = os.path.dirname(os.path.realpath(__file__))
    #print "file_dir:", file_dir
    #find parent dir
    parent_dir, _ = os.path.split(file_dir)
    #print "parent_dir: ", parent_dir, " _: ", _
    #test image folder path 
    src_path = os.path.join(parent_dir, SRC_DIR)
    #result image folder path
    dest_path = os.path.join(parent_dir, DEST_DIR)

    filenames = []
    for name in IMG_NAMES:
        base, ext = os.path.splitext(name)
        tempname = base + '-%s' + ext
        #print tempname
        filenames.append((os.path.join(src_path, name), os.path.join(dest_path, tempname)))
    
    #print "filename: ", filenames
    return filenames
"""
if __name__ == '__main__':
    filenames = get_filenames()
"""
