import torch
import pandas as pd
from tqdm import tqdm
import time
import os
import argparse

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize

from data_util import stadardize, getLookup
from util import getKernel, gp_mean_plot

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calcMSE(model, train_X, lookup, **kwargs):
    seed = kwargs.get("seed", None)
    m = kwargs.get("m", None) # size of validation set
    
    torch.manual_seed(seed + 1234)

    val_X = torch.rand(m, 2, dtype=torch.float64, device=device) * 2150
    val_Y = model.posterior(val_X).mean.detach()
    train_Y = model.posterior(train_X).mean.detach()

    train_loss = torch.mean((train_Y - lookup(train_X)) ** 2)
    val_loss = torch.mean((val_Y - lookup(val_X)) ** 2)
    return train_loss, val_loss

def calcR2(model, train_X, lookup, **kwargs):
    seed = kwargs.get("seed", None)
    m = kwargs.get("m", None) # size of validation set
    
    torch.manual_seed(seed + 1234)
    
    val_X = torch.rand(m, 2, dtype=torch.float64, device=device) * 2150
    val_Y = model.posterior(val_X).mean.detach()
    train_Y = model.posterior(train_X).mean.detach()

    # Calculate the mean of the training labels
    y_train_mean = lookup(train_X).mean()
    y_val_mean = lookup(val_X).mean()

    # Calculate total sum of squares (TSS)
    TSS_train = torch.sum((lookup(train_X) - y_train_mean) ** 2)
    TSS_val = torch.sum((lookup(val_X) - y_val_mean) ** 2)

    # Calculate residual sum of squares (RSS)
    RSS_train = torch.sum((train_Y - lookup(train_X)) ** 2)
    RSS_val = torch.sum((val_Y - lookup(val_X)) ** 2)

    # Calculate R^2
    R2_train = 1 - (RSS_train / TSS_train)
    R2_val = 1 - (RSS_val / TSS_val)

    return R2_train, R2_val


def main(args):
    seeds = 5
    n = args.n
    trait = args.env  # options: narea, sla, pn, ps
    transform = args.transform
    model = args.model_name
    kernel = getKernel(args.kernel, device=device)
    
    # create function to querying the environment
    table = getLookup(trait, transform)

    # directory to save results
    run_name = args.run_name
    if run_name == None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(f"./output/{run_name}"):
        os.mkdir(f"./output/{run_name}")
        
    #save args in a file
    with open(f"./output/{run_name}/args.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")
    
    def lookup(X):
        X_indices = X.long().cpu()
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)

    for seed in range(0, seeds):
        #query all random points
        torch.manual_seed(seed)  # setting seed
        full_train_X = torch.rand(n+10, 2, dtype=torch.float64, device=device) * 2150
        full_train_Y = lookup(full_train_X)

        # main loop
        _result = {"n": [], "train_loss": [], "val_loss": [], "best": []}
        for i in tqdm(range(n)):
            #select training points
            train_X, train_Y = full_train_X[:i+10], full_train_Y[:i+10]
            gp = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=kernel,
                outcome_transform=Standardize(1), 
                input_transform=Normalize(train_X.shape[-1]),
            )

            #this code is probably not necessary unless my inputs are in batches
#             gp._subset_batch_dict = {
#                 "mean_module.raw_constant": -1,
#                 "covar_module.raw_outputscale": -1,
#                 "covar_module.base_kernel.raw_lengthscale": -3,
#                 "likelihood.noise_covar.raw_noise": -2
#             }
            
            
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # calculate validation loss and record results
            kwargs = vars(args)
            kwargs["seed"] = seed
            train_loss, val_loss = calcR2(gp, train_X, lookup, **kwargs)
            _result["train_loss"].append(train_loss.item())
            _result["val_loss"].append(val_loss.item())
            _result["n"].append(i)
            _result["best"].append(max(train_Y).item())
            print(f"train loss: {train_loss}, validation loss: {val_loss}")

        # save result
        _result = pd.DataFrame(_result)
        _result.to_csv(
            f"./output/{run_name}/gp_{seed}.npy",
            encoding="utf8",
        )
        
        #save image and model here
        if args.plot_posterior: 
            gp_mean_plot(gp, f"./output/{run_name}/gp_{seed}.png", device=device)
            torch.save(gp.state_dict(), f"./output/{run_name}/gp_model_{seed}.pth")
       
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument("--n", type=int, default=300, help="Number of random points in training set")
    parser.add_argument("--m", type=int, default=3000, help="Number of random points in validation set")
    parser.add_argument("--transform", default="standardized", help="Transforming on the search space")
    parser.add_argument("--model_name", default="gp", help="Model to fit (default: SingleTaskGP)")
    parser.add_argument("--kernel", default=None, help="Model to fit (default: SingleTaskGP)")
    #parser.add_argument("--loss", default="MSE", help="=Type of loss function") #add this later
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    parser.add_argument("--run_name", default=None, help="Name of the folder to move outputs.")
    parser.add_argument("--plot_posterior", default=True, help="Do you want the posterior to be plotted at the end?")
    args = parser.parse_args()
    
    #set global device
    global device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    main(args)
