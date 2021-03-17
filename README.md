# OnePunchGAN2
GANs for Anime Keyframing

Update 1.2: The Python version has now been tested and works. In the spirit of tradition, I have also uploaded a ipynb with some changes made to it, OPGAN4. However, please note that currently and for the foreseeable future, the python version is the most up-to-date version of the code.

Update 1.1: With the update to milestone 2, the latest version of the google colab notebook is now OPGAN3.ipynb.

Note: The Python version is currently untested, as such, I have also uploaded the original Colab Notebook OPGAN.ipynb used to train and test the model for reference, though it is significantly messier. For a Colab notebook version of the code, please refer to the also uploaded OPGAN2.ipynb.

To use the github repo:

data_handler.py
Prepare your data. Unfortunately, the shell command that converts video into frames broke when transferring from Colab into Python, however, you can still run the ffmpeg command commented in data_handler.py via a terminal to execute the conversion, remember to fill in the placeholder variable names with the actual file name and locations. Alternatively, frames are available in the dataset that is linked seperately in this milestone submission.

Fill in the config data accordingly, and then run data_handler.py, it should produce the required chunks that the model takes in as input.

EDIT: With the update to Milestone 2, please remember to specify the number of the data task that you would like data_handler.py to perform. Additionally, to perform DAIN chunking, you will have to perform data task 3, then use DAIN on the even and odd output folders, followed by data task 4 on DAIN's output. The Github repo containing DAIN can be found at https://github.com/baowenbo/DAIN.

main.py
Fill in the config section as desired. You can toggle options for training and testing, set the location where the chunks for training and testing are stored, and other parameters to be used. Then, just run main.py and the model should take care of everything for you. There is a chance that the plotting code won't work, if that is the case, I recommend it be commented out. The plotting code will work in google Colab to display output.

EDIT: With the update to Milestone 2, please remember to specify the number of the training task. Unfortunately currently the testing code is only designed for testing the base model (which can be trained using training task 1)

The Colab notebook currently contains experimental code used to save outputs to file.

Enjoy and have fun.
