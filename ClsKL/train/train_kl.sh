#!/usr/bin/env bash

python train_kl.py --batch_size 24 \
    --cuda_id 4 \
    --net_type resnet \
    --depth 34
