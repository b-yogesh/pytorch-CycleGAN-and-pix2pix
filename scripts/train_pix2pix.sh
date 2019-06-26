set -ex
python train.py --dataroot ./datasets/night2day --name night2day_pix2pix --epoch_count 8 --display_id 0 --continue_train --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
