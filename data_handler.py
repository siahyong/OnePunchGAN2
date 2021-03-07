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

# Data processing type (1 - regular chunking, 2 - 5-frame chunking, 3 - DAIN splitting for pre-processing, 4 - DAIN chunking with the output from DAIN)
# To do DAIN processing, you'd run 3, run DAIN seperately on the two folders and their subfolders, then run 4
data_type = 1

# Locations of files for DAIN processing, is a two part process
source_folder = "CS236G_Project/test_frames" # Source for frames for DAIN processing
output_odd = "CS236G_Project/odd_frames" # Where you designated the odd frames output of the splitting (used for input into DAIN)
output_even = "CS236G_Project/even_frames" # Where you designated the even frames output of the splitting (used for input into DAIN)
DAIN_output_odd = '/content/gdrive/MyDrive/CS236G_Project/DAIN_odd' # Where you designated the odd frame outputs of DAIN (used for input into type 4 data processing)
DAIN_output_even = '/content/gdrive/MyDrive/CS236G_Project/DAIN_even' # Where you designated the even frame outputs of DAIN (used for input into type 4 data processing)

### END CONFIG ###

# Compatability code (translate between original colab notebook and established python file variables)
V_Chunk_Storage = CHUNK_OUTPUT_FOLDER
Frame_Storage = FRAME_OUTPUT_FOLDER

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

# Function to convert frames into 5 frame chunks that can be used for training
def image25chunk(folder,image_index, frame_gap = 1):
  a_name = '{}/{}/{}.png'.format(Frame_Storage, folder, f'{image_index:05}')
  b_name = '{}/{}/{}.png'.format(Frame_Storage, folder, f'{(image_index + frame_gap):05}')
  c_name = '{}/{}/{}.png'.format(Frame_Storage, folder, f'{(image_index + 2*frame_gap):05}')
  d_name = '{}/{}/{}.png'.format(Frame_Storage, folder, f'{(image_index + 3*frame_gap):05}')
  e_name = '{}/{}/{}.png'.format(Frame_Storage, folder, f'{(image_index + 4*frame_gap):05}')
  # print(a_name)
  a = np.array(Image.open(a_name))/255
  b = np.array(Image.open(b_name))/255
  c = np.array(Image.open(c_name))/255
  d = np.array(Image.open(d_name))/255
  e = np.array(Image.open(e_name))/255
  chunk = np.concatenate((a,b,c,d,e), axis = 1)
  chunk_im = Image.fromarray((chunk * 255).astype(np.uint8))
  chunk_im.save('{}/{}/{}.png'.format(V_Chunk_Storage, folder, f'{image_index:05}'))

# Function that splits files into folders based on parity of their index
def split_even_odd(folder, image_index):
  source = '{}/{}/{}.png'.format(source_folder, folder, f'{image_index:05}')
  destination = None
  if image_index % 2 == 0:
    destination = '{}/{}/{}.png'.format(output_even, folder, f'{(image_index//2):05}')
  else:
    destination = '{}/{}/{}.png'.format(output_odd, folder, f'{((image_index+1)//2):05}')
  shutil.copyfile(source, destination)

# Function that combines the output of DAIN into chunks that can be used for training
def make_dain_set(image_index, folder):
  even_output = DAIN_output_even
  odd_output = DAIN_output_odd

  dain_index = (image_index + 1) // 2
  pre = ''
  post = ''
  dain = ''
  real = ''
  if image_index%2 == 0:
    pre = '{}/{}/{}000.png'.format(even_output, folder, f'{dain_index:05}')
    post = '{}/{}/{}000.png'.format(even_output, folder, f'{(dain_index+1):05}')
    dain = '{}/{}/{}001.png'.format(even_output, folder, f'{dain_index:05}')
    real = '{}/{}/{}000.png'.format(odd_output, folder, f'{(dain_index+1):05}')
  elif image_index%2 == 1:
    pre = '{}/{}/{}000.png'.format(odd_output, folder, f'{dain_index:05}')
    post = '{}/{}/{}000.png'.format(odd_output, folder, f'{(dain_index+1):05}')
    dain = '{}/{}/{}001.png'.format(odd_output, folder, f'{dain_index:05}')
    real = '{}/{}/{}000.png'.format(even_output, folder, f'{(dain_index):05}')
  a = np.array(Image.open(pre))/255
  b = np.array(Image.open(post))/255
  c = np.array(Image.open(dain))/255
  d = np.array(Image.open(real))/255
  chunk = np.concatenate((a,b,c,d), axis = 1)
  chunk_im = Image.fromarray((chunk * 255).astype(np.uint8))
  chunk_im.save('{}/{}/{}.png'.format(CHUNK_OUTPUT_FOLDER, folder, f'{image_index:05}'))

if data_type == 1:
  # This bit here iterates through the list of provided subfolders and performs the frame to chunk function on them, for basic chunking
  for folder in folders:
      frame_count = len(list(glob.iglob("{}/{}/*.png".format(FRAME_OUTPUT_FOLDER, folder))))
      todo = frame_count - 2*frame_gap
      print("Commencing on frame folder {}".format(folder))
      for i in range(1, todo+1):
          image2chunk(FRAME_OUTPUT_FOLDER, folder, i, frame_gap)
      if i%50 == 0:
          print("Processed {} out of {} on todo list".format(i, todo))
      print("Finished on frame folder {}".format(folder))
elif data_type == 2:
  # This bit here iterates through the list of provided subfolders and performs the frame to chunk function on them, for 5 frame chunking
  for folder in folders:
    frame_count = len(list(glob.iglob("{}/{}/*.png".format(Frame_Storage, folder))))
    todo = frame_count - 4*frame_gap
    print("Commencing mega chunking on frame folder {}".format(folder))
    for i in range(1, todo+1):
      image25chunk(folder, i, frame_gap)
      if i%50 == 0:
        print("Processed {} out of {} on todo list".format(i, todo))
    print("Finished on frame folder {}".format(folder))
elif data_type == 3:
  # Splits dataset into even and odd for DAIN processing
  for folder in folders:
    frame_count = len(list(glob.iglob("{}/{}/*.png".format(source_folder, folder))))
    print("Commencing splitting on frame folder {}".format(folder))
    for i in range(1, frame_count+1):
      split_even_odd(folder, i)
      if i%50 == 0:
        print("Processed {} out of {} on todo list".format(i, frame_count))
    print("Finished on frame folder {}".format(folder))
elif data_type == 4:
  # Fuses DAIN outputs and original images into DAIN chunks
  for folder in folders:
    print("Beginning on folder {}".format(folder))
    frame_count = len(list(glob.iglob("{}/{}/*.png".format(Frame_Storage, folder))))
    todo = frame_count - 2
    for i in range(1, todo+1):
      make_dain_set(i, folder)
      if i%50 == 0:
        print("Completed {} out of {}".format(i, todo))
    print("Finished on folder {}".format(folder))
else:
  print("Invalid data task")
  
