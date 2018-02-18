#!/bin/bash


mkdir -p datasets

# download raw pictures
python3 tools/download.py

# convert raw pictures to .rec file
python3 tools/im2rec.py  --train-ratio=0.8 --test-ratio=0.2 --list datasets/captcha datasets/images
python3 tools/im2rec.py  --train-ratio=0.8 --test-ratio=0.2 --pack-label datasets datasets/images

