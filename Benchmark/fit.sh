#!/bin/bash
gpu=9
envs=("narea") #"sla" "pn" "ps")
kernels=("matern52" "matern32" "matern12" "rbf" "additive")
#acqs=("EI" "PI" "UCB-0.1" "UCB-0.2" "UCB-0.5")
n=300

# for env in ${envs[@]}; do
#     python main.py --env $env --n $n --acq random
#     #for acq in ${acqs[@]}; do
#     for kernel in ${kernels[@]}; do
#         python main.py --gpu $gpu --env $env --n $n --kernel $kernel --acq random#$acq 
#     done
#         #python main.py --gpu $gpu --env $env --n $n --model dnn --acq random 
#     #done

# done

trans=("box_cox" "log" "standardize")
kernels=("matern12")
for transform in ${trans[@]}; do
    for kernel in ${kernels[@]}; do
        env="narea"
        python gp_fit.py --gpu $gpu --env $env --n $n --kernel $kernel --run_name "${kernel}_prior_${transform}_${env}"
    done
done
