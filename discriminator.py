import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.pix2pix_model import Pix2PixModel
from models import networks
from util.visualizer import save_images
from util import html
import torch
from PIL import Image
import torchvision.transforms as transforms

def discriminate_sep(disc_model_path, image_a_path, image_b_path):

    # create the discriminator 
    netD = networks.define_D(input_nc=6, 
                             ndf=64, 
                             netD="basic",
                             n_layers_D=3, 
                             norm="batch", 
                             init_type="normal", 
                             init_gain=0.02, 
                             gpu_ids=[0])

    print('loading the model from %s' % disc_model_path)
    device = torch.device('cuda:0')
    state_dict = torch.load(disc_model_path, map_location=str(device))

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    netD.module.load_state_dict(state_dict)

    transform_list = []
    method=Image.BICUBIC

    # bring image to certain size
    load_size = 286
    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, method))

    # crop image to right dimensions
    crop_size = 256
    transform_list.append(transforms.RandomCrop(crop_size))

    # transform image to tensor
    transform_list += [transforms.ToTensor()]
    
    # normalize image
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform_data = transforms.Compose(transform_list)

    img_A_tensor = transform_data(Image.open(image_a_path).convert('RGB'))
    img_B_tensor = transform_data(Image.open(image_b_path).convert('RGB'))

    # adding dimension
    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_B_tensor = img_B_tensor.unsqueeze(0)

    real_A = img_A_tensor.to(device)
    real_B = img_B_tensor.to(device)

    real_AB = torch.cat((real_A, real_B), 1)

    pred = netD(real_AB.detach())

    print("Image {}: discriminator: {}".format(image_a_path, pred.mean()))

def discriminate(dataset, save_dir="./checkpoints/before2after_prepro"):

    # create the discriminator 
    netD = networks.define_D(input_nc=6, 
                             ndf=64, 
                             netD="basic",
                             n_layers_D=3, 
                             norm="batch", 
                             init_type="normal", 
                             init_gain=0.02, 
                             gpu_ids=[0])

    # load the weights from file
    load_filename = "latest_net_D.pth"
    load_path = os.path.join(save_dir, load_filename)
    
    print('loading the model from %s' % load_path)

    device = torch.device('cuda:0')

    state_dict = torch.load(load_path, map_location=str(device))

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    netD.module.load_state_dict(state_dict)

    # discriminate all data
    for i, data in enumerate(dataset):

        real_A = data["A"].to(device)
        real_B = data["B"].to(device)
        
        real_AB = torch.cat((real_A, real_B), 1)

        pred = netD(real_AB.detach())

        print("Image {}: discriminator: {}".format(i, pred.mean()))

def load_discriminator():
    pass

def load_image():
    pass      

if __name__ == '__main__':

    # opt = TestOptions().parse()  # get test options
    # dataset = create_dataset(opt)

    # discriminate(dataset)

    discriminate_sep(disc_model_path="./checkpoints/before2after_prepro/latest_net_D.pth", 
                     image_a_path="./datasets/before2after_prepro/A/train/train_0.jpg", 
                     image_b_path="./datasets/before2after_prepro/B/train/train_0.jpg")
