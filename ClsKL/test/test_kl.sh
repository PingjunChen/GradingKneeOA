#!/usr/bin/env bash

python test_kl.py --cuda-id 3 \
    --batch-size 16 \
    --phase "test" \
    --model_dir "../../data/ClsKLData/models/model_cmp/wm04/" \
    --best_model_name "0.700-0.344.pth"
