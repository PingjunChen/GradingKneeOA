#!/usr/bin/env bash

python train_kl.py --batch_size 24 \
    --cuda_id 6 \
    --net_type vgg \
    --depth 19 \
    --num_epoch 16 \
    --session 3
