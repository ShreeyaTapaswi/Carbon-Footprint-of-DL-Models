#!/usr/bin/env bash
# Experiment A on EuroSAT: all models, 3 seeds, identical hyperparameters.
for m in resnet18 resnet50 mobilenetv3_small efficientnet_b0 convnext_tiny vit_tiny deit_small; do
  for s in 0 1 2; do
    python train.py --model $m --seed $s --epochs 5
  done
done
