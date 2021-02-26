import torch
from torch import nn
from PIL import Image
import numpy as np
import glob as glob

### BEGIN CONFIG ###

# Location of input file, currently useless as shell ffmpeg does not work outside colab environment
INPUT_FILE = ''

# Main folder where generated frames are stored
FRAME_OUTPUT_FOLDER = ''

# Main folder for generated chunks to be stored
CHUNK_OUTPUT_FOLDER = ''

# Gap between frames for training
frame_gap = 1

# Subfolder names, if you don't have subfolders for your data, just put them all in one subfolder and fill in its name in the array below
folders = ['default']

### END CONFIG ###

# unfortunately, as I did all initial work on a google colab, the below line of code is currently broken
# I would recommend 
# %shell ffmpeg -i INPUT_FILE -vf 'select=gte(n\,1),setpts=PTS-STARTPTS,scale=704:480' '{FRAME_OUTPUT_FOLDER}/%05d.png'

# Function to convert frames into three frame chunks that will be used for training and testing
def image2chunk(FRAME_OUTPUT_FOLDER, folder, image_index, frame_gap = 1):
  a_name = '{}/{}/{}.png'.format(FRAME_OUTPUT_FOLDER, folder, f'{image_index:05}')
  b_name = '{}/{}/{}.png'.format(FRAME_OUTPUT_FOLDER, folder, f'{(image_index + frame_gap):05}')
  c_name = '{}/{}/{}.png'.format(FRAME_OUTPUT_FOLDER, folder, f'{(image_index + 2*frame_gap):05}')
  # print(a_name)
  a = np.array(Image.open(a_name))/255
  b = np.array(Image.open(b_name))/255
  c = np.array(Image.open(c_name))/255
  chunk = np.concatenate((a,c,b), axis = 1)
  chunk_im = Image.fromarray((chunk * 255).astype(np.uint8))
  chunk_im.save('{}/{}/{}.png'.format(CHUNK_OUTPUT_FOLDER, folder, f'{image_index:05}'))

# This bit here iterates through the list of provided subfolders and performs the frame to chunk function on them
for folder in folders:
    frame_count = len(list(glob.iglob("{}/{}/*.png".format(FRAME_OUTPUT_FOLDER, folder))))
    todo = frame_count - 2*frame_gap
    print("Commencing on frame folder {}".format(folder))
    for i in range(1, todo+1):
        image2chunk(FRAME_OUTPUT_FOLDER, folder, i, frame_gap)
    if i%50 == 0:
        print("Processed {} out of {} on todo list".format(i, todo))
    print("Finished on frame folder {}".format(folder))
