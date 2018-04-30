#!/bin/bash

# # model00
# python train_knee.py --device-id 3 \
#     --maxepoch 160 \
#     --weight_decay 0.0 \
#     --model-name kneedet00

# # model01
# python train_knee.py --device-id 2 \
#     --maxepoch 300 \
#     --weight_decay 5.0e-4 \
#     --model-name kneedet01


# model01
python train_knee.py --device-id 2 \
    --maxepoch 300 \
    --weight_decay 5.0e-4 \
    --model-name kneedet02
