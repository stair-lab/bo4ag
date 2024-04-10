import torch
import pandas as pd
from tqdm import tqdm
import time
import os

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getLookup(trait, transform=None):
    path = f"./data/{trait}_coh2.csv"
    lookup = pd.read_csv(path, header=0)

    lookup_tensor = torch.tensor(lookup.values, dtype=torch.float64)
    no_nan_lookup = torch.nan_to_num(lookup_tensor)
    no_nan_lookup[no_nan_lookup > 1] = 0
    
    #if transform...
    return no_nan_lookup

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

    # directory to save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(f"./output/{timestamp}")

    # create function to querying the environment
    table = getLookup(trait)
    
    def lookup(X):
        X_indices = X.long().cpu()
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)

    for seed in range(0, seeds):
        torch.manual_seed(seed)  # setting seed
        
        #query all random points
        full_train_X = torch.rand(n+10, 2, dtype=torch.float64, device=device) * 2150
        full_train_Y = lookup(full_train_X)
        print(full_train_X[:20])
        exit()

        # main loop
        _result = {"n": [], "train_loss": [], "val_loss": [], "best": []}
        for i in tqdm(range(n)):
            #select training points
            train_X, train_Y = full_train_X[:i+10], full_train_Y[:i+10]

            #fit the gp
            gp = SingleTaskGP(
                train_X,
                train_Y,
                outcome_transform=Standardize(1),
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

#         # save results
#         _result = pd.DataFrame(_result)
#         _result.to_csv(
#             f"./output/{timestamp}/gp_{seed}.npy",
#             encoding="utf8",
#         )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument("--n", type=int, default=300, help="Number of random points in training set")
    parser.add_argument("--m", type=int, default=3000, help="Number of random points in validation set")
    #parser.add_argument("--loss", default="MSE", help="=Type of loss function")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    args = parser.parse_args()
    main(args)
