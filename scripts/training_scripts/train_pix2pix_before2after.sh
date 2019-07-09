set -ex
python train.py \
    --dataroot ./datasets/before2after_prepro_paired \
    --name before2after_prepro \
    --display_id 0 \
    --model pix2pix \
    --netG unet_256 \
    --direction AtoB \
    --lambda_L1 100 \
    --dataset_mode aligned \
    --norm batch \
    --batch_size 100 \
