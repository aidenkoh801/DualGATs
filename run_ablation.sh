#!/bin/bash

# Common parameters
DATASET="MELD"
SEEDS=(100 101 102)  # Multiple seeds for reliability
EPOCHS=100

for seed in "${SEEDS[@]}"; do
    # Full Model
    python main.py --use_transformer True --use_cross_attention True \
        --dataset custom --epochs 150 --lr 1e-4 --dropout 0.4 --max_grad_norm 5.0  --batch_size 8 --gnn_layers 2 --weight_decay 0.01
    
    # Without Transformer
    python main.py --use_transformer False --use_cross_attention True \
        --dataset custom --epochs 100 --lr 1e-4 --dropout 0.6 --max_grad_norm 5.0  --batch_size 16 --gnn_layers 2 --weight_decay 0.01
    
    # Without Cross-Attention
    python main.py --use_transformer True --use_cross_attention False \
        --dataset custom --epochs 150 --lr 1e-4 --dropout 0.4 --max_grad_norm 5.0  --batch_size 8 --gnn_layers 2 --weight_decay 0.01
    
    # Baseline
    python main.py --use_transformer False --use_cross_attention False \
        --dataset custom --epochs 100 --lr 1e-4 --dropout 0.6 --max_grad_norm 5.0  --batch_size 16 --gnn_layers 2 --weight_decay 0.01
done
