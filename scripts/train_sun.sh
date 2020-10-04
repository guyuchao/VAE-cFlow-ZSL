#!/usr/bin/env bash
source /home/guyuchao/anaconda3/bin/activate ZSL

cd ..
python train_vaepriorflow.py --dataset SUN --niter 38000 --nSample 192