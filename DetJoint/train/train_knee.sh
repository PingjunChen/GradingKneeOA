#!/bin/bash

# model01
python train_knee.py --device-id 4 \
    --maxepoch 500 \
    --weight_decay 5.0e-4 \
    --batch_size 8 \
    --model-name kneedet06wd
