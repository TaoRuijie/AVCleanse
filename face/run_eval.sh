#!/bin/bash

python main.py \
--eval_trials /data08/VoxCeleb1/O_trials.txt \
--eval_list /data08/VoxCeleb1/O_list.txt \
--eval_path /data08/VoxCeleb1/ \
--save_path exps/debug \
--initial_model_v ../pretrain/V-Vox2.model \
--model_v res18 \
--eval \