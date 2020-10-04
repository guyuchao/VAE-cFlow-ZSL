#!/usr/bin/env bash
source /home/guyuchao/anaconda3/bin/activate ZSL

cd ..
python train_vaepriorflow.py --dataset AWA1 --niter 15000 --nSample 6000