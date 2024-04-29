import torch
import pandas as pd
import time
import argparse
from tqdm import tqdm
import os

from models.dnn import NNSurrogate
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# from botorch.utils import standardize
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, SpectralMixtureKernel
from botorch.optim import optimize_acqf
from botorch.acquisition import (
    qUpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qKnowledgeGradient,
)

from data_util import getLookup
# from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

# set device here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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


def getCoordTensor(size=2150):
    x, y = torch.arange(size), torch.arange(size)
    X, Y = torch.meshgrid(x, y)
    coordinates_tensor = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    return coordinates_tensor.to(device, torch.float64)

def calcLoss(model, train_X, lookup, **kwargs):
    seed = kwargs.get('seed', None)
    torch.manual_seed(seed) 
    m = 3000 #size of validation set
    
    val_X = torch.rand(m, 2, dtype=torch.float64, device=device) * 2150
    if isinstance(model, NNSurrogate):
        model.eval()
        val_Y = model(val_X)
        train_Y = model(train_X)
    else:
        val_Y = model.posterior(val_X).mean.detach()
        train_Y = model.posterior(train_X).mean.detach()
    
    train_loss = torch.mean((train_Y - lookup(train_X))**2)
    val_loss = torch.mean((val_Y - lookup(val_X))**2)
    return train_loss, val_loss

def runBO(args):
    n = args.n  
    trait = args.env
    transform = args.transform
    seeds = 5 
    
    acq_name = args.acq
    kernel = getKernel(args.kernel)
    model_name = args.model
    num_restarts = 128
    raw_samples = 128
  
    # create function for querying the environment
    table = getLookup(trait, )
    def lookup(X):
        X_indices = X.long().cpu() 
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)

    bounds = torch.stack(
        [torch.zeros(2).double(), torch.ones(2).double() * 2150]
    ).to(device, torch.float64)
    
    # create a directory to save results
    run_name = args.run_name
    if run_name == None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(f"./output/BO_runs/{trait}/{run_name}"):
        os.mkdir(f"./output/BO_runs/{trait}/{run_name}")

    # save args in a file
    with open(f"./output/BO_runs/{trait}/{run_name}/args.txt", "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    print(f"Running {trait}, {args.acq}-{args.kernel}...")
    for seed in range(0, seeds):  
        # collect random points as initial training points
        torch.manual_seed(seed) 
        train_X = torch.rand(10, 2, dtype=torch.float64, device=device) * 2150
        train_Y = lookup(train_X)

        # main bayes_opt training loop
        _result = {"n": [], "train_loss": [] , "val_loss": [], "best": []}
        for i in tqdm(range(n)):
            # set amd train the surrogate model
            if model_name == "gp":
                gp = SingleTaskGP(
                    train_X,
                    train_Y,
                    covar_module=kernel,
                    outcome_transform=Standardize(1),
                    input_transform=Normalize(train_X.shape[-1]),
                )

                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                model = gp
            elif model_name == "dnn": 
                '''
                Note: dnn surrogate can only work with random search
                '''
                model = NNSurrogate().to(device)
                model.fit(train_X, train_Y)
            else:
                print("Not a valid surrogate model")  # should also throw error
            
            #query from acquition function
            if acq_name == "random":
                new_X = torch.rand((i+1, 2), dtype=torch.float64, device=device,)[-1, :]#this is not working sequentially
                new_X = new_X * 2150  # adjust to match bounds
                new_X = new_X.reshape(1,2)
            else: 
                #all acquition functions which use `optimize_acqf` as the optimizer
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
            
            #calculate validation loss here 
            kwargs = {"seed": seed,}
            train_loss, val_loss = calcLoss(model, train_X, lookup, **kwargs)
            _result["train_loss"].append(train_loss.item())
            _result["val_loss"].append(val_loss.item())
            _result["n"].append(i)
            _result["best"].append(max(train_Y).item())
            print(f"train loss: {train_loss}, validation loss: {val_loss}, Best: {max(train_Y)}")

            # add new candidate
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])
            
         #save results
        _result = pd.DataFrame(_result)
        _result.to_csv(f"./output/BO_runs/{trait}/{run_name}/result_{seed}.npy", encoding="utf8",)

         #TODO: save the model
         

        
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument("--n", type=int, default=3000, help="Largest number of random points fit during training")
    parser.add_argument("--transform", default=None, help="Transforming on the search space")
    parser.add_argument("--kernel", default="rbf", help="Kernel function for the gaussian process")
    parser.add_argument("--acq", default="EI", help="Acquisition function")
    parser.add_argument("--model", default="gp", help="Surrogate Model (GP by default)")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    parser.add_argument("--run_name", default=None, help="Name of the folder to move outputs.")
    args = parser.parse_args()
    
    #set device globally
    global device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")
    
    #start your run
    runBO(args)
    return


main()
