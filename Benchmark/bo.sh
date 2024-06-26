#!/bin/bash
envs=("narea") # "sla" "pn" "ps")
kernels=("matern52" "matern32") # "matern12" "rbf" "spectral-10")
acqs=("EI" "PI") # "UCB-0.1" "UCB-0.2" "UCB-0.5")
n=300

for env in ${envs[@]}; do
    python main.py --env $env --n $n --acq random --run_name random
    for acq in ${acqs[@]}; do
        for kernel in ${kernels[@]}; do
            python main.py --env $env --n $n --kernel $kernel --acq $acq --run_name "${kernel}_${acq}"        
    	done
    done
done

# n=300
# python main.py --env pn --n $n --kernel matern52 --acq EI
# python main.py --env pn --n $n --kernel rbf --acq EI
# python main.py --env pn --n $n --kernel matern12 --acq EI

