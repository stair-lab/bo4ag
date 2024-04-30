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

from util.data_util import stadardize, getLookup
from util.util import getKernel, gp_mean_plot
from util.loss import calcR2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    seeds = 5
    n = args.n
    trait = args.env  # options: narea, sla, pn, ps
    transform = args.transform
    model = args.model_name
    kernel = getKernel(args.kernel, device=device)

    # create function to querying the environment
    table = getLookup(trait, transform)

    def lookup(X):
        X_indices = X.long().cpu()
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)

    # create a directory to save results
    run_name = args.run_name
    if run_name == None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(f"./output/{run_name}"):
        os.mkdir(f"./output/{run_name}")

    # save args in a file
    with open(f"./output/{run_name}/args.txt", "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    for seed in range(0, seeds):
        # query all random points
        torch.manual_seed(seed)  # setting seed
        full_train_X = torch.rand(n + 10, 2, dtype=torch.float64, device=device) * 2150
        full_train_Y = lookup(full_train_X)

        # main loop
        _result = {"n": [], "train_loss": [], "val_loss": [], "best": []}
        for i in tqdm(range(0, n+1, args.interval)):
            # select training points
            train_X, train_Y = full_train_X[: i + 10], full_train_Y[: i + 10]
            gp = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=kernel,
                outcome_transform=Standardize(1),
                input_transform=Normalize(train_X.shape[-1]),
            )

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # calculate validation loss and record results
            kwargs = vars(args)
            kwargs["seed"] = seed
            kwargs["device"] = device
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

        # save image and model here
        if args.plot_posterior:
            gp_mean_plot(gp, f"./output/{run_name}/gp_{seed}.png", device=device)
            torch.save(gp.state_dict(), f"./output/{run_name}/gp_model_{seed}.pth")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument("--n", type=int, default=300, help="Largest number of random points fit during training")
    parser.add_argument("--m", type=int, default=3000, help="Number of random points in validation set")
    parser.add_argument("--transform", default="standardized", help="Transforming on the search space")
    parser.add_argument("--model_name", default="gp", help="Model to fit (default: SingleTaskGP)")
    parser.add_argument("--kernel", default="rbf", help="Model to fit (default: SingleTaskGP)")
    # parser.add_argument("--loss", default="MSE", help="=Type of loss function") #add this later
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    parser.add_argument("--run_name", default=None, help="Name of the folder to move outputs.")
    parser.add_argument("--plot_posterior", type=int, default=True, help="Do you want the posterior to be plotted at the end?",)
    parser.add_argument("--interval", type=int, default=1, help="The number of iterations between each validation",)
    args = parser.parse_args()

    # set global device
    global device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    main(args)

#todo:
#fix file naming conventions
#fix the corresponding plotting code too