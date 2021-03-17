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
import glob as glob

### BEGIN CONFIG ###

# configuration parameters
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 200

# Bool to load previous saved state (base model)
pretrained = True

# Bool to load previous saved state (refinement model)
rn_pretrained = True

# Filename of previous state for loading base model
PREVIOUS_STATE_BASE = '../teenotters_base_7000.pth'

# Filename of previous state for loading refinement model
PREVIOUS_STATE_REFINE = '../teenotters_rn_7000.pth'

# Bool for saving of model
save_state = False

# How many iterations between checkpoints
CHECKPOINT_FREQ = 1000

# Folder where .pth checkpoints will be saved
SAVE_STATE_LOCATION = '..'

# A cute name for your checkpoints
SAVE_NAME = 'jojootters'

# The main folder where the chunks for training are saved (please include subfolders or the dataloader will crash)
TRAIN_CHUNK_OUTPUT_FOLDER = '../trainchunks'

# The main folder where the chunks for testing are saved (please include subfolders or the dataloader will crash)
TEST_CHUNK_OUTPUT_FOLDER = ''

# Do training?
training = True

# Do testing?
testing = False

# Number of epochs for training
n_epochs = 1

# Don't touch these, odds are your frames are RGB, so leave these as is
input_dim = 6
real_dim = 3
rn_input_dim = 9
mask_input_dim = 8
rn_mask_input_dim = 11

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

# How many images to count in validation
val_count_limit = 50

# What kind of training? (1 - base model, 2 - combined model, 3 - large model, 4 - DAIN model, 5 - mask model)
train_type = 6

### END CONFIG ###

# Compatibility stuff
save_location = SAVE_STATE_LOCATION
save_rate = CHECKPOINT_FREQ

if train_type == 5:
    input_dim = mask_input_dim
    rn_input_dim = rn_mask_input_dim

# Create generator and discriminator
gen = Generator(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Create refinement generator and discriminator
rn_gen = Generator(rn_input_dim, real_dim).to(device)
rn_gen_opt = torch.optim.Adam(rn_gen.parameters(), lr=lr)
rn_disc = Discriminator(rn_input_dim + real_dim).to(device)
rn_disc_opt = torch.optim.Adam(rn_disc.parameters(), lr=lr)

# Check if we are loading in a pretrained model, otherwise, initialise weights
if pretrained:
    loaded_state = torch.load(PREVIOUS_STATE_BASE, map_location=torch.device(device))
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc.load_state_dict(loaded_state["disc"])
    disc_opt.load_state_dict(loaded_state["disc_opt"])
else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

# Check if we are loading in a pretrained model, otherwise, initialise weights
if rn_pretrained:
  rn_loaded_state = torch.load(PREVIOUS_STATE_REFINE, map_location=torch.device(device))
  rn_gen.load_state_dict(rn_loaded_state["rn_gen"])
  rn_gen_opt.load_state_dict(rn_loaded_state["rn_gen_opt"])
  rn_disc.load_state_dict(rn_loaded_state["rn_disc"])
  rn_disc_opt.load_state_dict(rn_loaded_state["rn_disc_opt"])
else:
  rn_gen = rn_gen.apply(weights_init)
  rn_disc = rn_disc.apply(weights_init)    

# Prepare transform for preprocessing dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_gen = []
train_disc = []
rn_train_gen = []
rn_train_disc = []
val_gen = []
val_disc = []
rn_val_gen = []
rn_val_disc = []

mean_generator_loss = 0
mean_discriminator_loss = 0
rn_mean_generator_loss = 0
rn_mean_discriminator_loss = 0

val_count = 0
val_mean_gen_loss = 0
val_mean_disc_loss = 0
rn_val_mean_gen_loss = 0
rn_val_mean_disc_loss = 0

# Training Function, trains the generator using data from the training dataset
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

# Training function for combined model (base model + refinement model)
def train2(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    rn_mean_generator_loss = 0
    rn_mean_discriminator_loss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    # val_count_limit = 100

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(train_dataloader):
            #image_width = image.shape[3]
            #condition = image[:, :, :, :image_width // 2]
            #condition = nn.functional.interpolate(condition, size=target_shape)
            #real = image[:, :, :, image_width // 2:]
            #real = nn.functional.interpolate(real, size=target_shape)

            image_width = image.shape[3]
            pre = image[:, :, :, :image_width // 3]
            post = image[:, :, :, image_width // 3:2*image_width // 3]
            condition = torch.cat((pre, post), dim=1)
            real = image[:, :, :, 2*image_width // 3:]

            cur_batch_size = len(condition)
            condition = condition.to(device)
            real = real.to(device)

            dl, gl, rdl, rgl, fake, rn_fake = combo_train(condition, real)

            mean_generator_loss += gl
            mean_discriminator_loss += dl
            rn_mean_generator_loss += rgl
            rn_mean_discriminator_loss += rdl

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}, R-Generator (U-Net) loss: {rn_mean_generator_loss}, R-Discriminator loss: {rn_mean_discriminator_loss},")
                    train_gen.append(mean_generator_loss)
                    train_disc.append(mean_discriminator_loss)
                    rn_train_gen.append(rn_mean_generator_loss)
                    rn_train_disc.append(rn_mean_discriminator_loss)
                else:
                    print("Pretrained initial state")
                #show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(real, size=(3, 480, 704))
                show_tensor_images(fake, size=(3, 480, 704))
                show_tensor_images(rn_fake, size=(3, 480, 704))
                real_arr = np.array(make_grid(real.detach().cpu(), nrow = 5)).squeeze().transpose(1,2,0)
                real_image = Image.fromarray((real_arr * 255).astype(np.uint8))
                real_image.save('../trainoutputs/lotters_{}_real.png'.format(f'{cur_step:05}'))
                fake_arr = np.array(make_grid(fake.detach().cpu(), nrow = 5)).squeeze().transpose(1,2,0)
                fake_image = Image.fromarray((fake_arr * 255).astype(np.uint8))
                fake_image.save('../trainoutputs/lotters_{}_fake.png'.format(f'{cur_step:05}')) 
                rn_fake_arr = np.array(make_grid(rn_fake.detach().cpu(), nrow = 5)).squeeze().transpose(1,2,0)
                rn_fake_image = Image.fromarray((rn_fake_arr * 255).astype(np.uint8))
                rn_fake_image.save('../trainoutputs/lotters_{}_rn_fake.png'.format(f'{cur_step:05}'))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                rn_mean_generator_loss = 0
                rn_mean_discriminator_loss = 0

                val_count = 0
                val_mean_gen_loss = 0
                val_mean_disc_loss = 0
                rn_val_mean_gen_loss = 0
                rn_val_mean_disc_loss = 0
                for image, _ in tqdm(val_dataloader):
                  image_width = image.shape[3]
                  pre = image[:, :, :, :image_width // 3]
                  post = image[:, :, :, image_width // 3:2*image_width // 3]
                  condition = torch.cat((pre, post), dim=1)
                  real = image[:, :, :, 2*image_width // 3:]

                  cur_batch_size = len(condition)
                  condition = condition.to(device)
                  real = real.to(device)

                  dl, gl, rdl, rgl, fake, rn_fake = combo_train(condition, real, val=True, mean_count = val_count_limit)

                  val_mean_gen_loss += gl
                  val_mean_disc_loss += dl
                  rn_val_mean_gen_loss += rgl
                  rn_val_mean_disc_loss += rdl
                  
                  val_count += 1
                  if val_count >= val_count_limit:
                    break
                print("Validation Set Gen Loss: {}, Validation Set Disc Loss: {}, Validation Set R-Gen Loss: {}, Validation Set R-Disc Loss: {}".format(val_mean_gen_loss, val_mean_disc_loss, rn_val_mean_gen_loss, rn_val_mean_disc_loss))
                val_gen.append(val_mean_gen_loss)
                val_disc.append(val_mean_disc_loss)
                rn_val_gen.append(rn_val_mean_gen_loss)
                rn_val_disc.append(rn_val_mean_disc_loss)

                # You can change save_model to True if you'd like to save the model
            if cur_step % save_rate == 0:
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"{save_location}/{SAVE_NAME}_base_{cur_step}.pth")
                    torch.save({'rn_gen': rn_gen.state_dict(),
                        'rn_gen_opt': rn_gen_opt.state_dict(),
                        'rn_disc': rn_disc.state_dict(),
                        'rn_disc_opt': rn_disc_opt.state_dict()
                    }, f"{save_location}/{SAVE_NAME}_rn_{cur_step}.pth")
            cur_step += 1

# Condensor function to make everything neater, particuarly useful for 5-frame chunk
def combo_train(condition, real, val=False, mean_count = None): 
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
  # mean_discriminator_loss += disc_loss.item() / display_step
  # Keep track of the average generator loss
  # mean_generator_loss += gen_loss.item() / display_step

  rn_condition = torch.cat((fake, condition), dim=1)
  rn_disc_opt.zero_grad()
  with torch.no_grad():
    rn_fake = rn_gen(rn_condition)
  rn_disc_fake_hat = rn_disc(rn_fake.detach(), rn_condition) # Detach generator
  rn_disc_fake_loss = adv_criterion(rn_disc_fake_hat, torch.zeros_like(rn_disc_fake_hat))
  rn_disc_real_hat = rn_disc(real, rn_condition)
  rn_disc_real_loss = adv_criterion(rn_disc_real_hat, torch.ones_like(rn_disc_real_hat))
  rn_disc_loss = (rn_disc_fake_loss + rn_disc_real_loss) / 2
  rn_disc_loss.backward(retain_graph=True) # Update gradients
  rn_disc_opt.step() # Update optimizer

  rn_gen_opt.zero_grad()
  rn_gen_loss = get_gen_loss(rn_gen, rn_disc, real, rn_condition, adv_criterion, recon_criterion, lambda_recon)
  rn_gen_loss.backward() # Update gradients
  rn_gen_opt.step() # Update optimizer

  # rn_mean_discriminator_loss += rn_disc_loss.item() / display_step
  # Keep track of the average generator loss
  # rn_mean_generator_loss += rn_gen_loss.item() / display_step
  if val:
    return disc_loss.item() / mean_count, gen_loss.item() / mean_count, rn_disc_loss.item() / mean_count, rn_gen_loss.item() / mean_count, fake, rn_fake
  else:
    return disc_loss.item() / display_step, gen_loss.item() / display_step, rn_disc_loss.item() / display_step, rn_gen_loss.item() / display_step, fake, rn_fake

# Training function for mega-sized model (3 frames in between start and end frame)
def train3(save_model=False): 
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    rn_mean_generator_loss = 0
    rn_mean_discriminator_loss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    # val_count_limit = 100

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(train_dataloader):
            #image_width = image.shape[3]
            #condition = image[:, :, :, :image_width // 2]
            #condition = nn.functional.interpolate(condition, size=target_shape)
            #real = image[:, :, :, image_width // 2:]
            #real = nn.functional.interpolate(real, size=target_shape)

            # Note that for 5-chunks the frames are in order!
            image_width = image.shape[3]
            pre = image[:, :, :, :image_width // 5]
            post = image[:, :, :, 4*image_width // 5:]
            condition_2 = torch.cat((pre, post), dim=1)
            real_1 = image[:, :, :, image_width // 5: 2*image_width // 5]
            real_2 = image[:, :, :, 2*image_width // 5: 3*image_width // 5]
            real_3 = image[:, :, :, 3*image_width // 5: 4*image_width // 5]

            cur_batch_size = len(condition_2)
            pre = pre.to(device)
            post = post.to(device)
            condition_2 = condition_2.to(device)
            real_1 = real_1.to(device)
            real_2 = real_2.to(device)
            real_3 = real_3.to(device)

            # For the discriminator and generator, they will go through 3 updates, one for each of the 3 frames

            dl, gl, rdl, rgl, fake_2, rn_fake_2 = combo_train(condition_2, real_2)

            mean_generator_loss += gl
            mean_discriminator_loss += dl
            rn_mean_generator_loss += rgl
            rn_mean_discriminator_loss += rdl

            # mid = rn_fake_2.detach()
            mid = rn_fake_2
            condition_1 = torch.cat((pre, mid), dim=1)
            condition_3 = torch.cat((mid, post), dim=1)

            dl, gl, rdl, rgl, fake_1, rn_fake_1 = combo_train(condition_1, real_1)

            mean_generator_loss += gl
            mean_discriminator_loss += dl
            rn_mean_generator_loss += rgl
            rn_mean_discriminator_loss += rdl

            dl, gl, rdl, rgl, fake_3, rn_fake_3 = combo_train(condition_3, real_3)

            mean_generator_loss += gl
            mean_discriminator_loss += dl
            rn_mean_generator_loss += rgl
            rn_mean_discriminator_loss += rdl

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}, R-Generator (U-Net) loss: {rn_mean_generator_loss}, R-Discriminator loss: {rn_mean_discriminator_loss},")
                    train_gen.append(mean_generator_loss)
                    train_disc.append(mean_discriminator_loss)
                    rn_train_gen.append(rn_mean_generator_loss)
                    rn_train_disc.append(rn_mean_discriminator_loss)
                else:
                    print("Pretrained initial state")
                #show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(pre, size=(3, 480, 704))

                show_tensor_images(real_1, size=(3, 480, 704))
                show_tensor_images(fake_1, size=(3, 480, 704))
                show_tensor_images(rn_fake_1, size=(3, 480, 704))

                show_tensor_images(real_2, size=(3, 480, 704))
                show_tensor_images(fake_2, size=(3, 480, 704))
                show_tensor_images(rn_fake_2, size=(3, 480, 704))

                show_tensor_images(real_3, size=(3, 480, 704))
                show_tensor_images(fake_3, size=(3, 480, 704))
                show_tensor_images(rn_fake_3, size=(3, 480, 704))

                show_tensor_images(post, size=(3, 480, 704))

                mean_generator_loss = 0
                mean_discriminator_loss = 0
                rn_mean_generator_loss = 0
                rn_mean_discriminator_loss = 0

                val_count = 0
                val_mean_gen_loss = 0
                val_mean_disc_loss = 0
                rn_val_mean_gen_loss = 0
                rn_val_mean_disc_loss = 0
                for image, _ in tqdm(val_dataloader):
                  image_width = image.shape[3]
                  pre = image[:, :, :, :image_width // 5]
                  post = image[:, :, :, 4*image_width // 5:]
                  condition_2 = torch.cat((pre, post), dim=1)
                  real_1 = image[:, :, :, image_width // 5: 2*image_width // 5]
                  real_2 = image[:, :, :, 2*image_width // 5: 3*image_width // 5]
                  real_3 = image[:, :, :, 3*image_width // 5: 4*image_width // 5]

                  cur_batch_size = len(condition_2)
                  pre = pre.to(device)
                  post = post.to(device)
                  condition_2 = condition_2.to(device)
                  real_1 = real_1.to(device)
                  real_2 = real_2.to(device)
                  real_3 = real_3.to(device)

                  dl, gl, rdl, rgl, fake, rn_fake_2 = combo_train(condition_2, real_2, val=True, mean_count = val_count_limit)

                  val_mean_gen_loss += gl
                  val_mean_disc_loss += dl
                  rn_val_mean_gen_loss += rgl
                  rn_val_mean_disc_loss += rdl

                  # You might or might not want to detach rn_fake_2 to create the mid image
                  # It runs if we detach it, I don't know if we'll get a mem error if we don't
                  # mid = rn_fake_2.detach()
                  mid = rn_fake_2

                  # A side note on memory errors, it appears that Colab is giving me whichever GPU is free, depending on what
                  # I get, the speed of training varies by up to a factor of 2.5
                  # The memory error tends to occur when training is run multiple times, a quick fix is to factory reset the runtime
                  # Perhaps there is some kind of memory leak that I am not addressing

                  condition_1 = torch.cat((pre, mid), dim=1)
                  condition_3 = torch.cat((mid, post), dim=1)

                  dl, gl, rdl, rgl, fake, rn_fake_2 = combo_train(condition_1, real_1, val=True, mean_count = val_count_limit)

                  val_mean_gen_loss += gl
                  val_mean_disc_loss += dl
                  rn_val_mean_gen_loss += rgl
                  rn_val_mean_disc_loss += rdl

                  dl, gl, rdl, rgl, fake, rn_fake_2 = combo_train(condition_3, real_3, val=True, mean_count = val_count_limit)

                  val_mean_gen_loss += gl
                  val_mean_disc_loss += dl
                  rn_val_mean_gen_loss += rgl
                  rn_val_mean_disc_loss += rdl
                  
                  val_count += 1
                  if val_count >= val_count_limit:
                    break
                print("Validation Set Gen Loss: {}, Validation Set Disc Loss: {}, Validation Set R-Gen Loss: {}, Validation Set R-Disc Loss: {}".format(val_mean_gen_loss, val_mean_disc_loss, rn_val_mean_gen_loss, rn_val_mean_disc_loss))
                val_gen.append(val_mean_gen_loss)
                val_disc.append(val_mean_disc_loss)
                rn_val_gen.append(rn_val_mean_gen_loss)
                rn_val_disc.append(rn_val_mean_disc_loss)

                # You can change save_model to True if you'd like to save the model
            if cur_step % save_rate == 0:
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"/content/gdrive/MyDrive/{save_location}/orig_Vopgan_{cur_step}.pth")
                    torch.save({'rn_gen': rn_gen.state_dict(),
                        'rn_gen_opt': rn_gen_opt.state_dict(),
                        'rn_disc': rn_disc.state_dict(),
                        'rn_disc_opt': rn_disc_opt.state_dict()
                    }, f"/content/gdrive/MyDrive/{save_location}/rn_Vopgan_{cur_step}.pth")
            cur_step += 1

# Training function for refinement model that uses outputs from DAIN           
def dain_train(save_model=False): 
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    rn_mean_generator_loss = 0
    rn_mean_discriminator_loss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    val_count_limit = 100

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(train_dataloader):
            #image_width = image.shape[3]
            #condition = image[:, :, :, :image_width // 2]
            #condition = nn.functional.interpolate(condition, size=target_shape)
            #real = image[:, :, :, image_width // 2:]
            #real = nn.functional.interpolate(real, size=target_shape)

            image_width = image.shape[3]
            pre = image[:, :, :, :image_width // 4]
            post = image[:, :, :, image_width // 4:2*image_width // 4]
            dain = image[:, :, :, 2*image_width // 4:3*image_width // 4]
            rn_condition = torch.cat((pre, dain, post), dim=1)
            real = image[:, :, :, 3*image_width // 4:]

            cur_batch_size = len(rn_condition)
            rn_condition = rn_condition.to(device)
            real = real.to(device)
            dain = dain.to(device)

            rn_disc_opt.zero_grad()
            with torch.no_grad():
              rn_fake = rn_gen(rn_condition)
            rn_disc_fake_hat = rn_disc(rn_fake.detach(), rn_condition) # Detach generator
            rn_disc_fake_loss = adv_criterion(rn_disc_fake_hat, torch.zeros_like(rn_disc_fake_hat))
            rn_disc_real_hat = rn_disc(real, rn_condition)
            rn_disc_real_loss = adv_criterion(rn_disc_real_hat, torch.ones_like(rn_disc_real_hat))
            rn_disc_loss = (rn_disc_fake_loss + rn_disc_real_loss) / 2
            rn_disc_loss.backward(retain_graph=True) # Update gradients
            rn_disc_opt.step() # Update optimizer

            rn_gen_opt.zero_grad()
            rn_gen_loss = get_gen_loss(rn_gen, rn_disc, real, rn_condition, adv_criterion, recon_criterion, lambda_recon)
            rn_gen_loss.backward() # Update gradients
            rn_gen_opt.step() # Update optimizer

            rn_mean_generator_loss += rn_gen_loss.item() / display_step
            rn_mean_discriminator_loss += rn_disc_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}, R-Generator (U-Net) loss: {rn_mean_generator_loss}, R-Discriminator loss: {rn_mean_discriminator_loss},")
                    train_gen.append(mean_generator_loss)
                    train_disc.append(mean_discriminator_loss)
                    rn_train_gen.append(rn_mean_generator_loss)
                    rn_train_disc.append(rn_mean_discriminator_loss)
                else:
                    print("Pretrained initial state")
                #show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(real, size=(3, 480, 704))
                show_tensor_images(dain, size=(3, 480, 704))
                show_tensor_images(rn_fake, size=(3, 480, 704))

                mean_generator_loss = 0
                mean_discriminator_loss = 0
                rn_mean_generator_loss = 0
                rn_mean_discriminator_loss = 0

                val_count = 0
                val_mean_gen_loss = 0
                val_mean_disc_loss = 0
                rn_val_mean_gen_loss = 0
                rn_val_mean_disc_loss = 0
                for image, _ in tqdm(val_dataloader):
                  image_width = image.shape[3]
                  pre = image[:, :, :, :image_width // 4]
                  post = image[:, :, :, image_width // 4:2*image_width // 4]
                  dain = image[:, :, :, 2*image_width // 4:3*image_width // 4]
                  rn_condition = torch.cat((pre, dain, post), dim=1)
                  real = image[:, :, :, 3*image_width // 4:]

                  cur_batch_size = len(rn_condition)
                  rn_condition = rn_condition.to(device)
                  real = real.to(device)
                  dain = dain.to(device)

                  rn_disc_opt.zero_grad()
                  with torch.no_grad():
                    rn_fake = rn_gen(rn_condition)
                  rn_disc_fake_hat = rn_disc(rn_fake.detach(), rn_condition) # Detach generator
                  rn_disc_fake_loss = adv_criterion(rn_disc_fake_hat, torch.zeros_like(rn_disc_fake_hat))
                  rn_disc_real_hat = rn_disc(real, rn_condition)
                  rn_disc_real_loss = adv_criterion(rn_disc_real_hat, torch.ones_like(rn_disc_real_hat))
                  rn_disc_loss = (rn_disc_fake_loss + rn_disc_real_loss) / 2
                  rn_disc_loss.backward(retain_graph=True) # Update gradients
                  rn_disc_opt.step() # Update optimizer

                  rn_gen_opt.zero_grad()
                  rn_gen_loss = get_gen_loss(rn_gen, rn_disc, real, rn_condition, adv_criterion, recon_criterion, lambda_recon)
                  rn_gen_loss.backward() # Update gradients
                  rn_gen_opt.step() # Update optimizer

                  rn_val_mean_gen_loss += rn_gen_loss.item() / val_count_limit
                  rn_val_mean_disc_loss += rn_disc_loss.item() / val_count_limit
                  
                  val_count += 1
                  if val_count >= val_count_limit:
                    break
                print("Validation Set Gen Loss: {}, Validation Set Disc Loss: {}, Validation Set R-Gen Loss: {}, Validation Set R-Disc Loss: {}".format(val_mean_gen_loss, val_mean_disc_loss, rn_val_mean_gen_loss, rn_val_mean_disc_loss))
                val_gen.append(val_mean_gen_loss)
                val_disc.append(val_mean_disc_loss)
                rn_val_gen.append(rn_val_mean_gen_loss)
                rn_val_disc.append(rn_val_mean_disc_loss)

                # You can change save_model to True if you'd like to save the model
            if cur_step % save_rate == 0:
                if save_model:
                    torch.save({'rn_gen': rn_gen.state_dict(),
                        'rn_gen_opt': rn_gen_opt.state_dict(),
                        'rn_disc': rn_disc.state_dict(),
                        'rn_disc_opt': rn_disc_opt.state_dict()
                    }, f"/content/gdrive/MyDrive/{save_location}/Dopgan_{cur_step}.pth")
            cur_step += 1

def mask_train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    rn_mean_generator_loss = 0
    rn_mean_discriminator_loss = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    cur_step = 2000
    val_count_limit = 50

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(train_dataloader):
            #image_width = image.shape[3]
            #condition = image[:, :, :, :image_width // 2]
            #condition = nn.functional.interpolate(condition, size=target_shape)
            #real = image[:, :, :, image_width // 2:]
            #real = nn.functional.interpolate(real, size=target_shape)

            image_width = image.shape[3]
            pre = image[:, :, :, :image_width // 5]
            post = image[:, :, :, 2*image_width // 5:3*image_width // 5]
            pre_mask = image[:, 0:1, :, image_width // 5:2*image_width // 5]
            post_mask = image[:, 0:1, :, 3*image_width // 5:4*image_width // 5]
            condition = torch.cat((pre, pre_mask, post, post_mask), dim=1)
            real = image[:, :, :, 4*image_width // 5:]

            cur_batch_size = len(condition)
            condition = condition.to(device)
            real = real.to(device)

            dl, gl, rdl, rgl, fake, rn_fake = combo_train(condition, real)

            mean_generator_loss += gl
            mean_discriminator_loss += dl
            rn_mean_generator_loss += rgl
            rn_mean_discriminator_loss += rdl

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}, R-Generator (U-Net) loss: {rn_mean_generator_loss}, R-Discriminator loss: {rn_mean_discriminator_loss},")
                    train_gen.append(mean_generator_loss)
                    train_disc.append(mean_discriminator_loss)
                    rn_train_gen.append(rn_mean_generator_loss)
                    rn_train_disc.append(rn_mean_discriminator_loss)
                else:
                    print("Pretrained initial state")
                #show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(real, size=(3, 480, 704))
                show_tensor_images(fake, size=(3, 480, 704))
                show_tensor_images(rn_fake, size=(3, 480, 704))
                real_arr = np.array(make_grid(real.detach().cpu(), nrow = 5)).squeeze().transpose(1,2,0)
                real_image = Image.fromarray((real_arr * 255).astype(np.uint8))
                real_image.save('../trainoutputs/motters_{}_real.png'.format(f'{cur_step:05}'))
                fake_arr = np.array(make_grid(fake.detach().cpu(), nrow = 5)).squeeze().transpose(1,2,0)
                fake_image = Image.fromarray((fake_arr * 255).astype(np.uint8))
                fake_image.save('../trainoutputs/motters_{}_fake.png'.format(f'{cur_step:05}')) 
                rn_fake_arr = np.array(make_grid(rn_fake.detach().cpu(), nrow = 5)).squeeze().transpose(1,2,0)
                rn_fake_image = Image.fromarray((rn_fake_arr * 255).astype(np.uint8))
                rn_fake_image.save('../trainoutputs/motters_{}_rn_fake.png'.format(f'{cur_step:05}'))

                mean_generator_loss = 0
                mean_discriminator_loss = 0
                rn_mean_generator_loss = 0
                rn_mean_discriminator_loss = 0

                val_count = 0
                val_mean_gen_loss = 0
                val_mean_disc_loss = 0
                rn_val_mean_gen_loss = 0
                rn_val_mean_disc_loss = 0
                for image, _ in tqdm(val_dataloader):
                  image_width = image.shape[3]
                  pre = image[:, :, :, :image_width // 5]
                  post = image[:, :, :, 2*image_width // 5:3*image_width // 5]
                  pre_mask = image[:, 0:1, :, image_width // 5:2*image_width // 5]
                  post_mask = image[:, 0:1, :, 3*image_width // 5:4*image_width // 5]
                  condition = torch.cat((pre, pre_mask, post, post_mask), dim=1)
                  real = image[:, :, :, 4*image_width // 5:]

                  cur_batch_size = len(condition)
                  condition = condition.to(device)
                  real = real.to(device)

                  dl, gl, rdl, rgl, fake, rn_fake = combo_train(condition, real, val=True, mean_count = val_count_limit)

                  val_mean_gen_loss += gl
                  val_mean_disc_loss += dl
                  rn_val_mean_gen_loss += rgl
                  rn_val_mean_disc_loss += rdl
                  
                  val_count += 1
                  if val_count >= val_count_limit:
                    break
                print("Validation Set Gen Loss: {}, Validation Set Disc Loss: {}, Validation Set R-Gen Loss: {}, Validation Set R-Disc Loss: {}".format(val_mean_gen_loss, val_mean_disc_loss, rn_val_mean_gen_loss, rn_val_mean_disc_loss))
                val_gen.append(val_mean_gen_loss)
                val_disc.append(val_mean_disc_loss)
                rn_val_gen.append(rn_val_mean_gen_loss)
                rn_val_disc.append(rn_val_mean_disc_loss)

                # You can change save_model to True if you'd like to save the model
            if cur_step % save_rate == 0:
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict()
                    }, f"{save_location}/{SAVE_NAME}_base_{cur_step}.pth")
                    torch.save({'rn_gen': rn_gen.state_dict(),
                        'rn_gen_opt': rn_gen_opt.state_dict(),
                        'rn_disc': rn_disc.state_dict(),
                        'rn_disc_opt': rn_disc_opt.state_dict()
                    }, f"{save_location}/{SAVE_NAME}_rn_{cur_step}.pth")
            cur_step += 1

# Testing Function, draws a sample from the testing dataset and runs the generator on it
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

# Does what it says, makes frames for the jojo opening, 7 out of 8 frames synthesized
def make_jojo():
    frame_count = len(list(glob.iglob("../jojo_frames/jojo/*.png")))
    indice_count = (frame_count - 1) // 4
    for i in range(indice_count):
        print("{} breads eaten out of {}".format(i+1, indice_count))
        frame1 = np.array(Image.open('../jojo_frames/jojo/{}.png'.format(f'{(4*i + 1):05}')))/255
        frame2 = np.array(Image.open('../jojo_frames/jojo/{}.png'.format(f'{(4*i + 5):05}')))/255

        frame1 = np.expand_dims(frame1, axis = 0)
        frame2 = np.expand_dims(frame2, axis = 0)
        ba = torch.FloatTensor(np.transpose(frame1, (0, 3, 1, 2))).to(device)
        aa = torch.FloatTensor(np.transpose(frame2, (0, 3, 1, 2))).to(device)
        mid_con = torch.cat((ba, aa), dim=1)
        mid_con_m = gen(mid_con).detach()
        midput = rn_gen(torch.cat((mid_con_m, mid_con), dim=1)).detach()

        first_q = torch.cat((ba, midput), dim=1)
        first_q_m = gen(first_q).detach()
        fqput = rn_gen(torch.cat((first_q_m, first_q), dim=1)).detach()

        third_q = torch.cat((midput, aa), dim=1)
        third_q_m = gen(third_q).detach()
        tqput = rn_gen(torch.cat((third_q_m, third_q), dim=1)).detach()

        #first_e = torch.cat((ba, fqput), dim=1)
        #first_e_m = gen(first_e).detach()
        #fieput = rn_gen(torch.cat((first_e_m, first_e), dim=1)).detach()

        #second_e = torch.cat((fqput, midput), dim=1)
        #second_e_m = gen(second_e).detach()
        #seeput = rn_gen(torch.cat((second_e_m, second_e), dim=1)).detach()

        #third_e = torch.cat((midput, tqput), dim=1)
        #third_e_m = gen(third_e).detach()
        #theput = rn_gen(torch.cat((third_e_m, third_e), dim=1)).detach()

        #fourth_e = torch.cat((tqput, aa), dim=1)
        #fourth_e_m = gen(fourth_e).detach()
        #foeput = rn_gen(torch.cat((fourth_e_m, fourth_e), dim=1)).detach()
        
        #frames = [ba, fieput, fqput, seeput, midput, theput, tqput, foeput]
        frames = [ba, fqput, midput, tqput]
        count = 1
        for thing in frames:
            pre_arr = np.array(thing.cpu()).squeeze().transpose(1,2,0)
            pre_image = Image.fromarray((pre_arr * 255).astype(np.uint8))
            pre_image.save('../newer_jojo/{}.png'.format(f'{(4*i + count):05}'))
            count += 1

# Execute training
if training:
    dataset = torchvision.datasets.ImageFolder('{}'.format(TRAIN_CHUNK_OUTPUT_FOLDER), transform=transform)
    dataset_len = len(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [dataset_len - dataset_len//5, dataset_len//5])
    if train_type == 1:
        train(save_model=save_state)
    elif train_type == 2:
        train2(save_model=save_state)
    elif train_type == 3:
        train3(save_model=save_state)
    elif train_type == 4:
        dain_train(save_model=save_state)
    elif train_type == 5:
        mask_train(save_model=save_state)
    elif train_type == 6:
        make_jojo()
    else:
        print("Invalid training type")
        
# Execute testing
if testing:
    test_dataset = torchvision.datasets.ImageFolder('{}'.format(TEST_CHUNK_OUTPUT_FOLDER), transform=transform)
    test_generator()
