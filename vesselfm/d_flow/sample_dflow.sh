#!/bin/bash

python sample.py \
    --ckpt_path runs/fm/model-20.pt \
    --num_samples 10 \
    --mask_folder data/d_drand/d_drand \
    --out_folder data/d_flow \
    --class_cond \
    --num_classes 2
    # --no_npy \
    # --nifti \
    # --overview \