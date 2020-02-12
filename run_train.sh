#!/bin/bash

# REPO_PATH=${REPO_PATH:-~/Desktop/Python/PPCAN_new}
# cd $REPO_PATH

if [ ! -d "results" ]; then
	mkdir results
fi

MODE="train"

NUM_SIMULATIONS=5

REACH_CONVERGENCE_1="False"
MSE_EPSILON_1=1e-4
TRAIN_EPOCHS_1=1
TRAIN_ITERS_LEGIT_1=30000

REACH_CONVERGENCE_2="False"
ACC_EPSILON_2=1e-3
TRAIN_EPOCHS_2=1
TRAIN_ITERS_ADV_2=30000

REACH_CONVERGENCE_3="False"
MSE_EPSILON_3=1e-4
ACC_EPSILON_3=1e-3
TRAIN_EPOCHS_3=40
TRAIN_ITERS_LEGIT_3=500
TRAIN_ITERS_ADV_3=8000
LOSS_TYPE=2
ALPHA=1

CONV_DEPTH=16
SNR_LEGIT_TRAIN=10
SNR_ADV_TRAIN=5

LEARN_RATE=0.0001
BATCH_SIZE=64

DATASET="cifar"


python -u main.py --mode $MODE \
                  --num_simulations $NUM_SIMULATIONS \
                  --reach_convergence_1 $REACH_CONVERGENCE_1 \
                  --mse_epsilon_1 $MSE_EPSILON_1 \
                  --train_epochs_1 $TRAIN_EPOCHS_1 \
                  --train_iters_legit_1 $TRAIN_ITERS_LEGIT_1 \
                  \
                  --reach_convergence_2 $REACH_CONVERGENCE_2 \
                  --acc_epsilon_2 $ACC_EPSILON_2 \
                  --train_epochs_2 $TRAIN_EPOCHS_2 \
                  --train_iters_adv_2 $TRAIN_ITERS_ADV_2 \
                  \
                  --reach_convergence_3 $REACH_CONVERGENCE_3 \
                  --mse_epsilon_3 $MSE_EPSILON_3 \
                  --acc_epsilon_3 $ACC_EPSILON_3 \
                  --train_epochs_3 $TRAIN_EPOCHS_3 \
                  --train_iters_legit_3 $TRAIN_ITERS_LEGIT_3 \
                  --train_iters_adv_3 $TRAIN_ITERS_ADV_3 \
                  --loss_type $LOSS_TYPE \
                  --alpha $ALPHA \
                  \
                  --conv_depth $CONV_DEPTH \
                  --snr_legit_train $SNR_LEGIT_TRAIN \
                  --snr_adv_train $SNR_ADV_TRAIN \
                  \
                  --learn_rate $LEARN_RATE \
                  --batch_size $BATCH_SIZE \
                  --dataset $DATASET \
