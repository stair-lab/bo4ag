#!/bin/bash
gpu=0
envs=("narea") #"sla" "pn" "ps")
kernels=("matern52" "matern32" "matern12" "rbf" "spectral-10")
acqs=("EI" "PI" "UCB-0.1" "UCB-0.2" "UCB-0.5")
n=30

for env in ${envs[@]}; do
    python main.py --env $env --n $n --acq random
    for acq in ${acqs[@]}; do
        for kernel in ${kernels[@]}; do
            python main.py --gpu $gpu --env $env --n $n --kernel $kernel --acq $acq 
        done
    done
done
