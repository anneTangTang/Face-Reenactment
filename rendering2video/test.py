import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import ImageDataset
from models import *

assert torch.cuda.is_available(), "cuda is not available!"

# specify
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, help="root folder containing recons and eye folder")
parser.add_argument("--target", type=str, help="root folder of target")
args = parser.parse_args()

models_path = os.path.join(args.target, "train_result", "models")
save_path = os.path.join(args.root, "generated")

# Initialize generator
generator = GeneratorFork(in_channels=6, out_channels=3).cuda()
# Load pre-trained models
generator.load_state_dict(torch.load(os.path.join(models_path, "generator_130.pth")))
generator.eval()
print("Load Generator succeed!")

########################
#      Testing
########################

total = 0
for img in os.listdir(os.path.join(args.root, "recons")):
    total += 1

batch_size = 10
test_dataloader = DataLoader(
    ImageDataset(root=args.root, mode="test", total=total), batch_size=batch_size, shuffle=False, num_workers=0,
)

for i, batch in enumerate(test_dataloader):
    real_A = Variable(batch["input"].type(torch.cuda.FloatTensor))
    fake_B = generator(real_A)  # Variable. (-1, 1)
    fake_B = fake_B.data.cpu().numpy()  # np. (10, 3, h, w). (-1, 1)
    fake_B = ((fake_B * 0.5) + 0.5) * 255  # np. (10, 3, h, w). (0.0, 255.0)
    fake_B = np.uint8(fake_B)  # np. (10, 3, h, w). (0, 255)
    fake_B = np.swapaxes(fake_B, 1, 3)  # np. (10, w, h, 3). (0, 255)
    fake_B = np.swapaxes(fake_B, 1, 2)  # np. (10, h, w, 3). (0, 255)
    for j in range(batch_size):
        img_name = "%05d.jpg" % (i * batch_size + j + 1)
        img = fake_B[j]  # np. (h, w, 3). (0, 255)
        img = Image.fromarray(img)
        img.save(os.path.join(save_path, img_name))
