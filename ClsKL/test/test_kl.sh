#!/usr/bin/env bash

python test_kl.py --cuda-id 0 \
    --batch-size 16 \
    --model_dir "../../data/ClsKLData/models/model_best/vgg19/vgg-19-SGD-1" \
    --best_model_name 9-0.677-0.696-0.475.pth
