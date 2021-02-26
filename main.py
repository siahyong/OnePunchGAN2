import torch
from torch import nn
from PIL import Image
import numpy as np
import torch.nn.functional as F
from model import Generator, Discriminator
from util import weights_init, get_gen_loss, show_tensor_images
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from skimage import color

### BEGIN CONFIG ###

# configuration parameters
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 200

# Bool to load previous saved state
pretrained = False

# Filename of previous state for loading
PREVIOUS_STATE = ''

# Bool for saving of model
save_state = False

# How many iterations between checkpoints
CHECKPOINT_FREQ = 2000

# Folder where .pth checkpoints will be saved
SAVE_STATE_LOCATION = ''

# A cute name for your checkpoints
SAVE_NAME = 'babyotters'

# The main folder where the chunks for training are saved (please include subfolders or the dataloader will crash)
TRAIN_CHUNK_OUTPUT_FOLDER = ''

# The main folder where the chunks for testing are saved (please include subfolders or the dataloader will crash)
TEST_CHUNK_OUTPUT_FOLDER = ''

# Do training?
training = True

# Do testing?
testing = True

# Number of epochs for training
n_epochs = 2

# Don't touch these, odds are your frames are RGB, so leave these as is
input_dim = 6
real_dim = 3

# How often should the generator show its work
display_step = 500

# Batch size, caution, numbers larger than one could devour all your RAM
batch_size = 1

# learning rate
lr = 0.0002

# Deprecated, ignore
target_shape = 256

# Device for training, change to 'cpu' if CUDA unavailable. Note, has not been tested on 'cpu' setting
device = 'cuda'

### END CONFIG ###

gen = Generator(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

if pretrained:
    loaded_state = torch.load(PREVIOUS_STATE)
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc.load_state_dict(loaded_state["disc"])
    disc_opt.load_state_dict(loaded_state["disc_opt"])
else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Training Function
def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    val_count_limit = 100

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(train_dataloader):

            image_width = image.shape[3]
            pre = image[:, :, :, :image_width // 3]
            post = image[:, :, :, image_width // 3:2*image_width // 3]
            condition = torch.cat((pre, post), dim=1)
            real = image[:, :, :, 2*image_width // 3:]

            condition = condition.to(device)
            real = real.to(device)

            ### Update discriminator ###
            disc_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(), condition) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                #show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(real, size=(3, 480, 704))
                show_tensor_images(fake, size=(3, 480, 704))

                mean_generator_loss = 0
                mean_discriminator_loss = 0

                val_count = 0
                val_mean_gen_loss = 0
                val_mean_disc_loss = 0
                for image, _ in tqdm(val_dataloader):
                  image_width = image.shape[3]
                  pre = image[:, :, :, :image_width // 3]
                  post = image[:, :, :, image_width // 3:2*image_width // 3]
                  condition = torch.cat((pre, post), dim=1)
                  real = image[:, :, :, 2*image_width // 3:]

                  condition = condition.to(device)
                  real = real.to(device)

                  disc_opt.zero_grad() # Zero out the gradient before backpropagation
                  with torch.no_grad():
                      fake = gen(condition)
                  disc_fake_hat = disc(fake.detach(), condition) # Detach generator
                  disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
                  disc_real_hat = disc(real, condition)
                  disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
                  disc_loss = (disc_fake_loss + disc_real_loss) / 2

                  ### Update generator ###
                  gen_opt.zero_grad()
                  gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)

                  val_mean_gen_loss += gen_loss.item() / val_count_limit
                  val_mean_disc_loss += disc_loss.item() / val_count_limit
                  
                  val_count += 1
                  if val_count >= val_count_limit:
                    break
                print("Validation Set Gen Loss: {}, Validation Set Disc Loss: {}".format(val_mean_gen_loss, val_mean_disc_loss))

                # You can change save_model to True if you'd like to save the model
            if cur_step % CHECKPOINT_FREQ == 0:
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"{SAVE_STATE_LOCATION}/{SAVE_NAME}_{cur_step}.pth")
            cur_step += 1

# Testing Function
def test_generator(test_count_limit = 300):
  test_count = 0
  test_mean_gen_loss = 0
  test_mean_disc_loss = 0
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  for image, _ in tqdm(test_dataloader):
    image_width = image.shape[3]
    pre = image[:, :, :, :image_width // 3]
    post = image[:, :, :, image_width // 3:2*image_width // 3]
    condition = torch.cat((pre, post), dim=1)
    real = image[:, :, :, 2*image_width // 3:]

    condition = condition.to(device)
    real = real.to(device)

    disc_opt.zero_grad() # Zero out the gradient before backpropagation
    with torch.no_grad():
        fake = gen(condition)
    disc_fake_hat = disc(fake.detach(), condition) # Detach generator
    disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
    disc_real_hat = disc(real, condition)
    disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    gen_opt.zero_grad()
    gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)

    test_mean_gen_loss += gen_loss.item() / test_count_limit
    test_mean_disc_loss += disc_loss.item() / test_count_limit

    if test_count % 10 == 0:
      show_tensor_images(real, size=(3, 480, 704))
      show_tensor_images(fake, size=(3, 480, 704))

    test_count += 1
    if test_count >= test_count_limit:
      break
  print("Test Set Gen Loss: {}, Test Set Disc Loss: {}".format(test_mean_gen_loss, test_mean_disc_loss))

if training:
    dataset = torchvision.datasets.ImageFolder('{}'.format(TRAIN_CHUNK_OUTPUT_FOLDER), transform=transform)
    dataset_len = len(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [dataset_len - dataset_len//5, dataset_len//5])
    train(save_model=save_state)

if testing:
    test_dataset = torchvision.datasets.ImageFolder('{}'.format(TEST_CHUNK_OUTPUT_FOLDER), transform=transform)
    test_generator()
