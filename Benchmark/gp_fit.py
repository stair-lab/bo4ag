import torch
import pandas as pd
from tqdm import tqdm
import time
import os
import argparse

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize
from scipy import stats

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stadardize(data):
    mean = data.mean()
    std = data.std()    
    standardized_data = (data - mean) / std
    return standardized_data

def getLookup(trait, transform=None):
    path = f"./data/{trait}_coh2.csv"
    lookup = pd.read_csv(path, header=0)

    # Replace nans with zero
    lookup_tensor = torch.tensor(lookup.values, dtype=torch.float64)
    no_nan_lookup = torch.nan_to_num(lookup_tensor)
    no_nan_lookup[no_nan_lookup > 1] = 0
    
    # Make zeros, non-zero
    mask = no_nan_lookup == 0
    no_nan_lookup[mask] = 10e-6
    
    # Apply input transforms
    if transform == "log":
        log_data = torch.log(no_nan_lookup)
        return stadardize(log_data) #standardize
    elif transform == "box_cox":
        flat_lookup = no_nan_lookup.reshape(-1)
        fitted_data, fitted_lambda = stats.boxcox(flat_lookup)
        unflat_lookup = no_nan_lookup.reshape(no_nan_lookup.shape)
        
        print((unflat_lookup - no_nan_lookup))
        print(no_nan_lookup)
        exit()
        return stadardize(unflat_lookup) #standardize
    elif transform is not None:
        print(f"{transform} is not a valid input tranform...")
    
    return stadardize(no_nan_lookup) #standardize

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
    trait = args.env  # options are narea, sla, pn, ps
    transform = args.transform
    
    # create function to querying the environment
    table = getLookup(trait, transform)

    # directory to save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(f"./output/{timestamp}")
    
    #save args in a file
    with open(f"./output/{timestamp}/args.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")
    
    def lookup(X):
        X_indices = X.long().cpu()
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)

    for seed in range(0, seeds):
        torch.manual_seed(seed)  # setting seed
        
        #query all random points
        full_train_X = torch.rand(n+10, 2, dtype=torch.float64, device=device) * 2150
        full_train_Y = lookup(full_train_X)

        # main loop
        _result = {"n": [], "train_loss": [], "val_loss": [], "best": []}
        for i in tqdm(range(n)):
            #select training points
            train_X, train_Y = full_train_X[:i+10], full_train_Y[:i+10]

            #fit the gp
            gp = SingleTaskGP(
                train_X,
                train_Y,
                #outcome_transform=Standardize(1), moved standardize to the data loading part! (refer to getLookup())
                input_transform=Normalize(train_X.shape[-1]),
            )
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
            # print(f"train loss: {train_loss}, validation loss: {val_loss}")

        # save result
        _result = pd.DataFrame(_result)
        _result.to_csv(
            f"./output/{timestamp}/gp_{seed}.npy",
            encoding="utf8",
        )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument("--n", type=int, default=300, help="Number of random points in training set")
    parser.add_argument("--m", type=int, default=3000, help="Number of random points in validation set")
    parser.add_argument("--transform", default=None, help="Transforming on the search space")
    #parser.add_argument("--loss", default="MSE", help="=Type of loss function")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    args = parser.parse_args()
    
    #set global device
    global device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    main(args)
