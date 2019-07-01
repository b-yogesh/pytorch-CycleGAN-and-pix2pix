set -ex
python test.py --dataroot ./datasets/night2day --name night2day_pix2pix --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch
