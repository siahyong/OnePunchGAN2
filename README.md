# OnePunchGAN2
GANs for Anime Keyframing

NOTE: The Python version is currently untested, as such, I have also uploaded the original Colab Notebook OPGAN.ipynb used to train and test the model for reference, though it is significantly messier. For a Colab notebook version of the code, please refer to the also uploaded OPGAN2.ipynb.

To use the github repo:

data_handler.py
Prepare your data. Unfortunately, the shell command that converts video into frames broke when transferring from Colab into Python, however, you can still run the ffmpeg command commented in data_handler.py via a terminal to execute the conversion, remember to fill in the placeholder variable names with the actual file name and locations. Alternatively, frames are available in the dataset that is linked seperately in this milestone submission.

Fill in the config data accordingly, and then run data_handler.py, it should produce the required chunks that the model takes in as input.

main.py
Fill in the config section as desired. You can toggle options for training and testing, set the location where the chunks for training and testing are stored, and other parameters to be used. Then, just run main.py and the model should take care of everything for you. There is a chance that the plotting code won't work, if that is the case, I recommend it be commented out. The plotting code will work in google Colab to display output.

The Colab notebook currently contains experimental code used to save outputs to file.

Enjoy and have fun.
