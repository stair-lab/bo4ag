import torch
import pandas as pd
import time
import argparse
from tqdm import tqdm

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
# from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

# set device here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getLookup(trait):
    path = f"./data/{trait}_coh2.csv"
    lookup = pd.read_csv(path, header=0)

    # fix formatting
    lookup_tensor = torch.tensor(lookup.values, dtype=torch.float64)
    no_nan_lookup = torch.nan_to_num(lookup_tensor)
    no_nan_lookup[no_nan_lookup > 1] = 0
    return no_nan_lookup
    

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

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", help="Environment to run search.")
    parser.add_argument(
        "--kernel", default="rbf", help="Kernel function for the gaussian process"
    )
    parser.add_argument("--acq", default="EI", help="Acquisition function")
    parser.add_argument("--n", type=int, default=300, help="Number of iterations")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id to run the job")
    args = parser.parse_args()
    
    #set device globally
    global device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")

#     if args.acq == "random":
#         runRandom(args)
#     else:
#         runBO(args)
    runBO(args)
    return

def runRandom(args):
    n = args.n  # replace this
    trait = args.env
    seeds = 5  # consider replacing this
    acq_name = "random"
    lookup = getLookup(args.env)

    # check the lookup table
    assert not torch.isnan(torch.sum(lookup))
    assert not torch.isinf(torch.sum(lookup))

    print(f"Running {trait}, random search...")
    for seed in range(0, seeds):  # seed one is already run and stuff
        torch.manual_seed(seed)
        tic = time.perf_counter()  # start time

        train_X = torch.rand(10, 2, dtype=torch.float64, device=device) * (
            lookup.shape[0] - 1
        )
        train_Y = torch.tensor(
            [
                lookup[int(train_X[i][0]), int(train_X[i][1])]
                for i in range(0, len(train_X))
            ],
            dtype=torch.float64,
            device=device,
        ).reshape(-1, 1)

        _result = []
        for i in tqdm(range(n)):
            new_X = torch.rand(
                (1, 2),
                dtype=torch.float64,
                device=device,
            )
            new_X = new_X * 2150  # adjust to match bounds
            new_Y = torch.tensor(
                [lookup[int(new_X[0][0]), int(new_X[0][1])]],
                dtype=torch.float64,
                device=device,
            ).reshape(-1, 1)
            # add new candidate
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

            # end timer and add
            toc = time.perf_counter()  # end time
            _result.append([new_Y[0][0].item(), toc - tic, new_X[0]])
            

        # save all your queries
        torch.save(train_X, f"./output/{trait}/botorch{acq_name}_X_{seed}.npy")
        torch.save(train_Y, f"./output/{trait}/botorch{acq_name}_Y_{seed}.npy")

        # organize the list to have running best
        best = [0, 0, 0]  # format is [time, best co-heritabilty]
        botorch_result = []
        for i in _result:
            if i[0] > best[0]:
                best = i
            botorch_result.append(
                [best[0], i[1], best[2]]
            )  # append [best so far, current time]
        print("Best From Run: ", best)

        # store results
        botorch_result = pd.DataFrame(
            botorch_result, columns=["Best", "Time", "Candidate"]
        )
        botorch_result.to_csv(
            f"./output/{trait}/botorch{acq_name}_result_{seed}.npy", encoding="utf8"
        )  # store botorch search results

        # print full time
        toc = time.perf_counter()  # end time
        print("BoTorch Took ", (toc - tic), "seconds")
    return

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
    val_Y = model.posterior(val_X).mean.detach()
    train_Y = model.posterior(train_X).mean.detach()
    
    train_loss = torch.mean((train_Y - lookup(train_X))**2)
    val_loss = torch.mean((val_Y - lookup(val_X))**2)
    return train_loss, val_loss

def runBO(args):
    num_restarts = 128
    raw_samples = 128
    n = args.n  # replace this
    trait = args.env
    seeds = 5  # consider replacing this
    acq_name = args.acq
    kernel_name = args.kernel
    kernel = getKernel(kernel_name)
  
    # create function for querying the environment
    table = getLookup(args.env)
    def lookup(X):
        X_indices = X.long().cpu() 
        Y = table[X_indices[:, 0], X_indices[:, 1]].reshape(-1, 1)
        return Y.to(device, torch.float64)

    bounds = torch.stack(
        [torch.zeros(2).double(), torch.ones(2).double() * (table.shape[0] - 1)]
    ).to(device, torch.float64)
    
    # check the lookup table
    assert not torch.isnan(torch.sum(table))
    assert not torch.isinf(torch.sum(table))

    print(f"Running {trait}, {args.acq}-{kernel_name}...")
    for seed in range(0, seeds):  # seed one is already run and stuff
        tic = time.perf_counter()  # start time

        # collect random points as training points
        torch.manual_seed(seed)  # setting seed
        train_X = torch.rand(10, 2, dtype=torch.float64, device=device) * 2150
        train_Y = lookup(train_X)

        # main bayes_opt training loop
        _result = []
        _curve = {"n": [], "train_loss": [] , "val_loss": []}
        run_name = f"{acq_name}_{kernel_name}"
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
            
            if acq_name == "random":
                run_name = f"{acq_name}"
                new_X = torch.rand((1, 2), dtype=torch.float64, device=device,)
                new_X = new_X * 2150  # adjust to match bounds
            else: 
                #all acquition functions which use `optimize_acqf` as the optimizer
                if "UCB" in acq_name:
                    _, beta = acq_name.split("-")
                    beta = float(beta)
                    acq = qUpperConfidenceBound(gp, beta=beta)
                elif acq_name == "EI":  # working
                    acq = qExpectedImprovement(gp, best_f=max(train_Y))
                elif acq_name == "PI":  # working
                    acq = qProbabilityOfImprovement(gp, best_f=max(train_Y))
                elif acq_name == "KG":  # working
                    num_restarts = 10
                    acq = qKnowledgeGradient(gp)
                elif acq_name == "MES":
                    continue
                # TODO
                #                 x1_values = np.linspace(0, 2150, 2150)
                #                 x2_values = np.linspace(0, 2150, 2150)
                #                 x1, x2 = np.meshgrid(x1_values, x2_values)
                #                 candidates = np.vstack([x1.ravel(), x2.ravel()]).T
                #                 candidates = torch.from_numpy(candidates).to(device=device)
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
            train_loss, val_loss = calcLoss(gp, train_X, lookup, **kwargs)
            _curve["train_loss"].append(train_loss.item())
            _curve["val_loss"].append(val_loss.item())
            _curve["n"].append(i)
            
            # add new candidate
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

            # end timer and add
            toc = time.perf_counter()  # end time
            _result.append([new_Y[0][0].item(), toc - tic, new_X[0]])

        # save all your queries
        torch.save(train_X, f"./output/{trait}/botorch{run_name}_X_{seed}.npy")
        torch.save(train_Y, f"./output/{trait}/botorch{run_name}_Y_{seed}.npy")
        
        #save learning curve
        curve_df = pd.DataFrame(_curve)
        curve_df.to_csv(f"./output/{trait}/botorch{run_name}_curve_{seed}.npy",
            encoding="utf8",)
        
        # organize the list to have running best
        best = [0, 0, 0]  # format is [time, best co-heritabilty]
        botorch_result = []
        for i in _result:
            if i[0] > best[0]:
                best = i
            botorch_result.append(
                [best[0], i[1], best[2]]
            )  # append [best so far, current time]
        print("Best From Run: ", best)

        # store results
        botorch_result = pd.DataFrame(
            botorch_result, columns=["Best", "Time", "Candidate"]
        )
        botorch_result.to_csv(
            f"./output/{trait}/botorch{run_name}_result_{seed}.npy",
            encoding="utf8",
        )  # store botorch search results

        # print full time
        toc = time.perf_counter()  # end time
        print("BoTorch Took ", (toc - tic), "seconds")
        
    return

main()
