import argparse
import datetime
import os
import sys
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .datasets import *
from .models import *

assert torch.cuda.is_available(), "cuda is not available!"

# specify
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="", help="root folder containing train or test data")

parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=130, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument(
    "--sample_interval", type=int, default=1000, help="interval between sample of images from generator"
)
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
opt = parser.parse_args()

# preprocess
img_height = img_width = 256
save_path = os.path.join(opt.root, "train_result")
models_path = os.path.join(save_path, "models")
sample_path = os.path.join(save_path, "sample")
os.makedirs(save_path, exist_ok=False)
os.makedirs(models_path, exist_ok=False)
os.makedirs(sample_path, exist_ok=False)

total = 0
for img in os.listdir(os.path.join(opt.root, "origin")):
    total += 1

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
lambda_GAN = 50

# Calculate output size of image discriminator (PatchGAN)
patch = (1, img_height // 16, img_width // 16)

# Initialize generator and discriminator
generator = GeneratorFork(in_channels=6, out_channels=3).cuda()
discriminator = Discriminator(in_channels=9).cuda()
# generator = nn.DataParallel(generator, device_ids=opt.device_ids)
# discriminator = nn.DataParallel(discriminator, device_ids=opt.device_ids)

if opt.epoch != 1:
    # Load pre-trained models
    generator.load_state_dict(torch.load(os.path.join(models_path, f"generator_{opt.epoch - 1}.pth")))
    discriminator.load_state_dict(torch.load(os.path.join(models_path, f"discriminator_{opt.epoch - 1}.pth")))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
train_dataloader = DataLoader(
    ImageDataset(root=opt.root, mode="train", total=total),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset(root=opt.root, mode="val", total=total), batch_size=10, shuffle=True, num_workers=opt.n_cpu,
)

# Tensor type
Tensor = torch.cuda.FloatTensor


def sample_images(batches_done):
    """Save generated samples from the validation set."""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["input"].type(Tensor))
    real_B = Variable(imgs["ground_truth"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((fake_B.data, real_B.data), -2)  # (-1, 1)
    img_sample = (img_sample + 1.0) / 2.0  # (0, 1)
    save_image(
        img_sample, os.path.join(sample_path, f"{batches_done}.png"), nrow=5, normalize=False,
    )


########################
#      Training
########################
prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs + 1):
    for i, batch in enumerate(train_dataloader):
        batches_done = (epoch - 1) * len(train_dataloader) + i + 1
        # Model inputs
        real_A = Variable(batch["input"].type(Tensor))
        real_B = Variable(batch["ground_truth"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        mask = Variable(batch["mask"].type(Tensor))  # (3,h,w)
        loss_pixel = torch.pow(torch.abs(fake_B - real_B), 1)  # (3, h, w)
        loss_pixel = torch.mean(loss_pixel * mask)

        # Total loss
        loss_G = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = lambda_GAN * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # ------------------
        #  Log Progress
        # ------------------

        # Determine approximate time left
        batches_left = opt.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            f"\r[Epoch {epoch}/{opt.n_epochs}] [Batch {i + 1}/{len(train_dataloader)}] [D loss: {loss_D.item()}] "
            f"[G loss: {loss_G.item()}, pixel: {loss_pixel.item()}, adv: {loss_GAN.item()}] ETA: {time_left}"
        )

        # If at sample interval, save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and (epoch % opt.checkpoint_interval == 0 or epoch == opt.n_epochs):
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(models_path, f"generator_{epoch}.pth"))
        torch.save(
            discriminator.state_dict(), os.path.join(models_path, f"discriminator_{epoch}.pth"),
        )
