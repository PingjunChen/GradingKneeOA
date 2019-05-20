#!/usr/bin/env bash

python test_kl.py --cuda-id 6 \
    --batch-size 16 \
    --phase "test" \
    --model_dir "../../data/ClsKLData/models/model_cmp/wm04/" \
    --best_model_name "0.697-0.344.pth"
