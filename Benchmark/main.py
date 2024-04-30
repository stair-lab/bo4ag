import os
import time
import argparse
from tqdm import tqdm

import torch
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    qUpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qKnowledgeGradient,
)

from data_util import getLookup


def getKernel(kernel_name):
    covar_module = None
    if kernel_name == "matern52":
        covar_module = ScaleKernel(MaternKernel(nu=5 / 2, ard_num_dims=2))
    elif kernel_name == "matern32":
        covar_module = ScaleKernel(MaternKernel(nu=3 / 2, ard_num_dims=2))
    elif kernel_name == "matern12":
        covar_module = ScaleKernel(MaternKernel(nu=1 / 2, ard_num_dims=2))
    elif kernel_name == "rbf":
        covar_module = ScaleKernel(RBFKernel())
    elif "spectral" in kernel_name:
        _, num_mixtures = kernel_name.split("-")
        num_mixtures = int(num_mixtures)
        covar_module = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=2)
    else:
        print("Not a valid kernel")  # should also throw error
    return covar_module


def runBO(args):    
    n = args.n  
    trait = args.env
    seeds = 5 
    
    acq_name = args.acq
    kernel_name = args.kernel
    kernel = getKernel(args.kernel)
    transform = args.transform 
    num_restarts = 128
    raw_samples = 128

    # get lookup environment
    table = getLookup(trait, transform)

    def lookup(X):
        X_indices = X.long().cpu()
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)
    
    bounds = torch.tensor([[0,0],[2150, 2150]]).to(device, torch.float64)
    
    # create a directory to save results
    run_name = args.run_name
    if run_name == None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(f"./output/bo/{trait}/{run_name}"):
        os.mkdir(f"./output/bo/{trait}/{run_name}")

    # save args in a file
    with open(f"./output/bo/{trait}/{run_name}/args.txt", "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")
    
    print(f"Running {trait}, {args.acq}-{kernel_name}...")
    for seed in range(0, seeds):  
        # collect random points as training points
        torch.manual_seed(seed) 
        train_X = torch.rand(10, 2, dtype=torch.float64, device=device) * (2149)
        train_Y = lookup(train_X)

        # main bayes_opt training loop
        _result = {"n": [], "Best": []}
        _result["n"], _result["Best"] = zip(*[(x, torch.max(train_Y[:x]).item()) for x in range(1,11)])
        _result["n"], _result["Best"] = list(_result["n"]), list(_result["Best"])
        for i in tqdm(range(n)):
            gp = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=kernel,
                outcome_transform=Standardize(1),
                input_transform=Normalize(train_X.shape[-1]),
            )

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # select acquisition function
            if acq_name == "random":
                new_X = torch.rand((i+1, 2), dtype=torch.float64, device=device,)[-1, :]
                new_X = new_X * 2149 
                new_X = new_X.reshape(1,2)
            else:
                if "UCB" in acq_name:
                    _, beta = acq_name.split("-")
                    beta = float(beta)
                    acq = qUpperConfidenceBound(gp, beta=beta)
                elif acq_name == "EI":  
                    acq = qExpectedImprovement(gp, best_f=max(train_Y))
                elif acq_name == "PI": 
                    acq = qProbabilityOfImprovement(gp, best_f=max(train_Y))
                elif acq_name == "KG":  
                    num_restarts = 10
                    acq = qKnowledgeGradient(gp)
                else:
                    print(f"{acq_name} is not a valid acquisition function")

                new_X, acq_value = optimize_acqf(
                    acq,
                    q=1,
                    bounds=bounds,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
            new_Y = lookup(new_X)

            # add new candidate
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

            # stored results
            _result["n"].append(i+10)
            _result["Best"].append(max(train_Y).item())
            
            #store stuff here

        #save result
        _result = pd.DataFrame(_result)
        _result.to_csv(
            f"./output/bo/{trait}/{run_name}/result_{seed}.npy",
            encoding="utf8",
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument("--kernel", default="rbf", help="Kernel function for the gaussian process")
    parser.add_argument("--acq", default="EI", help="Acquisition function")
    parser.add_argument("--n", type=int, default=300, help="Number of iterations")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    parser.add_argument("--run_name", default=None, help="Name of the folder to move outputs.")
    parser.add_argument("--transform", default=None, help="Transforming on the search space")
    args = parser.parse_args()
    
    #set device globally
    global device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")
      
#     import wandb
#     run = wandb.init(project="bo4ag", config=args)
#     wandb_config_as_python_dict = dict(wandb.config)
#     wandb.finish()
    runBO(args)
    
    return
main()