#!/bin/bash


mkdir -p datasets/images
mkdir -p datasets/images/train
mkdir -p datasets/images/test

# download raw pictures
python3 tools/download.py

# convert raw pictures to .rec file
python3 tools/im2rec.py  --train-ratio=1 --test-ratio=0 --pack-label datasets/train datasets/images/train
python3 tools/im2rec.py  --train-ratio=1 --test-ratio=0 --pack-label datasets/test datasets/images/test

