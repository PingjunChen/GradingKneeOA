#!/bin/bash

python train_knee.py --device-id 2 --model-name det_models00 &
python train_knee.py --device-id 7 --model-name det_models01 &
