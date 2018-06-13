#!/bin/bash
#
for i in 1 2 3 4 5 6 7;
do
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
    source activate initial-keras
    python cnn_growing.py $i "cnn_growing_"$i"_lr1e-6_decay1e-6_largedense_5x5filter_nesterov_dataaug_norm_nodropouts_shuffle_150x150_incfilters.txt"
done
