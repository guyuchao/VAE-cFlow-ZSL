#!/usr/bin/env bash
source /home/guyuchao/anaconda3/bin/activate ZSL

cd ..
python train_vaepriorflow.py --dataset CUB --niter 35000 --nSample 600