#!/bin/bash

envs=("narea" "sla" "pn" "ps")
kernels=("matern52" "matern32" "matern12" "rbf" "spectral-10")
acqs=("EI" "PI" "UCB" "KG" "UCB-0.1" "UCB-0.2" "UCB-0.5")
n=300

for env in ${envs[@]}; do
    python main.py --env $env --n $n --acq random
    for acq in ${acqs[@]}; do
        for kernel in ${kernels[@]}; do
            python main.py --env $env --n $n --kernel $kernel --acq $acq 
        done
    done
done

#lets check the loop is working first