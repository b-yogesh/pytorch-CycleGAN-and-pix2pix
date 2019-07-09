set -ex
python test.py \
    --dataroot ./datasets/before2after_prepro_paired \
    --name before2after_prepro \
    --model pix2pix \
    --netG unet_256 \
    --direction AtoB \
    --dataset_mode aligned \
    --norm batch
