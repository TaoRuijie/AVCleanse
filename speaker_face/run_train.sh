#!/bin/bash

python main.py \
--train_list /data08/VoxCeleb2/train_mini.txt \
--train_path /data08/VoxCeleb2 \
--eval_trials /data08/VoxCeleb1/O_trials.txt \
--eval_list /data08/VoxCeleb1/O_list.txt \
--eval_path /data08/VoxCeleb1/ \
--save_path exps/debug \
--n_class 5994 \
--lr 0.001 \
--lr_decay 0.97 \
--scale_a 30 \
--margin_a 0.2 \
--model_a ecapa1024 \
--scale_v 64 \
--margin_v 0.4 \
--model_v res18 \
--test_step 1 \
--max_epoch 100 \
--train