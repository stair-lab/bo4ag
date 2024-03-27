import numpy as np
import torch 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import sys
import argparse
from tqdm import tqdm

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize #this might make things take longer to calculate??? 
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, SpectralMixtureKernel


from botorch.optim import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound 
from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement, qKnowledgeGradient#, qPredictiveEntropySearch

#set device here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)

def getLookup(trait):
#     names = {"narea": "Narea", "sla": "SLA", "ps": "PLSR_SLA_Sorghum", "pn": "FS_PLSR_Narea"}
#     trait = trait.lower()
#     trait = names[trait]
    
#     base_path = "/dfs/scratch0/ruhana/GenCor"
#     path = f"{base_path}/Table-Env/table_env/envs/csv_files/{trait}_Full_Analysis.csv"
#     csv_mat = np.genfromtxt(path, delimiter=',' )
#     x_starter, x_end, y_starter, y_end, lookup = csv_mat[0,1], csv_mat[0,-1], csv_mat[1,0], csv_mat[-1,0], csv_mat[1:, 1:]
    path = f"/lfs/turing2/0/ruhana/gptransfer/Benchmark/data/{trait}_coh2.csv"
    lookup = pd.read_csv(path, header=0)
    
    ##fix formatting
    lookup_tensor = torch.tensor(lookup.values, dtype=torch.float64)
    no_nan_lookup = torch.nan_to_num(lookup_tensor)
    return lookup

def getKernel(kernel_name):
    covar_module = None
    if kernel_name == "matern52": covar_module = ScaleKernel(MaternKernel(nu=5/2, ard_num_dims=2)) 
    elif kernel_name == "matern32": covar_module = ScaleKernel(MaternKernel(nu=3/2, ard_num_dims=2)) 
    elif kernel_name == "matern12": covar_module = ScaleKernel(MaternKernel(nu=1/2, ard_num_dims=2)) 
    elif kernel_name == "rbf": covar_module =  ScaleKernel(RBFKernel())
    elif "spectral" in kernel_name: 
        _, num_mixtures = kernel_name.split("-")
        num_mixtures = int(num_mixtures)
        covar_module = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=2)
    else:
        print("Not a valid kernel") #should also throw error
    return covar_module

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env', help='Environment to run search.')
    parser.add_argument('--kernel', default='rbf',
                        help='Kernel function for the gaussian process')
    parser.add_argument('--acq', default='EI', help='Acquisition function')
    parser.add_argument('--n', type=int, default=300, help='Number of iterations')
    
    args = parser.parse_args()
    
    if args.acq == "random":
        runRandom(args)
    else:
        runBO(args)
    return 

def runRandom(args):
    n = args.n #replace this
    trait = args.env
    seeds = 3 #consider replacing this
    acq_name = "random"
    
    #get lookup environment
    lookup = getLookup(args.env)
    ub, lb = 2150, 0

    #check the lookup table
    #not sure about this lookup table...
    assert np.isnan(np.sum(lookup)) == False 
    assert np.isinf(np.sum(lookup)) == False
    
    ##main bayes_opt training loop
    train_X = torch.empty((0, 2), dtype=torch.float64, device=device)
    train_Y = torch.empty((0, 1), dtype=torch.float64, device=device)

    for seed in range(0,seeds): #seed one is already run and stuff
        torch.manual_seed(seed)
        tic = time.perf_counter() #start time
        
        _result = []
        for i in tqdm(range(n)):
            new_X = torch.rand((1, 2), dtype=torch.float64, device=device,)
            new_X = new_X * (ub - lb) + lb #adjust to match bounds
            new_Y = torch.tensor([lookup[int(new_X[0][0]), int(new_X[0][1])]], 
                                 dtype=torch.float64, 
                                 device=device).reshape(-1, 1)
            #add new candidate
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

            #end timer and add
            toc = time.perf_counter() #end time
            _result.append([new_Y[0][0].item(), toc - tic, new_X[0]])

        #save all your queries
        torch.save(train_X, f"./output/{trait}/botorch{acq_name}_X_{seed}.npy")
        torch.save(train_Y, f"./output/{trait}/botorch{acq_name}_Y_{seed}.npy")

        #organize the list to have running best
        best = [0,0,0] # format is [time, best co-heritabilty]
        botorch_result = []
        for i in _result:
            if i[0] > best[0]:
                best = i
            botorch_result.append([best[0], i[1], best[2]]) # append [best so far, current time]
        print("Best From Run: ", best)

        #store results
        botorch_result = pd.DataFrame(botorch_result, columns=["Best", "Time", "Candidate"])
        botorch_result.to_csv(f"./output/{trait}/botorch{acq_name}_result_{seed}.npy", encoding='utf8') #store botorch search results

        #print full time
        toc = time.perf_counter() #end time
        print("BoTorch Took ", (toc-tic) ,"seconds")
    
    return 

def runBO(args):
    num_restarts = 128  
    raw_samples = 128
    n = args.n #replace this
    trait = args.env
    seeds = 5 #consider replacing this
    kernel_name = args.kernel
    kernel = getKernel(kernel_name)
    
    #get lookup environment
    lookup = getLookup(args.env)
    bounds = torch.stack([torch.zeros(2).double(), torch.ones(2).double() * (lookup.shape[0]-1)]).to(device, torch.float64)

    #check the lookup table
    assert np.isnan(np.sum(lookup)) == False 
    assert np.isinf(np.sum(lookup)) == False

    for seed in range(0,seeds): #seed one is already run and stuff
        tic = time.perf_counter() #start time

        ##collect random points as training points
        torch.manual_seed(seed) #setting seed
        train_X = torch.rand(10, 2, dtype=torch.float64, device=device) * (lookup.shape[0]-1)
        train_Y = torch.tensor([lookup[int(train_X[i][0]), int(train_X[i][1])] for i in range(0, len(train_X))], 
                               dtype=torch.float64, 
                               device=device)
        train_Y = train_Y.reshape(-1, 1)

        ##main bayes_opt training loop
        _result = []
        for i in tqdm(range(n)):
            gp = SingleTaskGP(
                train_X, train_Y, 
                covar_module = kernel,
                outcome_transform=Standardize(1), 
                input_transform=Normalize(train_X.shape[-1])
            )

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            #select acquisition function
            acq_name = args.acq
            if acq_name == "UCB":
                acq = qUpperConfidenceBound(gp, beta=0.1)
            elif acq_name == "EI": #working
                acq = qExpectedImprovement(gp, best_f=max(train_Y))
            elif acq_name == "PI": #working
                acq = qProbabilityOfImprovement(gp, best_f=max(train_Y))
            elif acq_name == "KG": #working
                acq = qKnowledgeGradient(gp)  
            else:
                print(f"{acq_name} is not a valid acquisition function")
            
            new_X, acq_value = optimize_acqf(
                acq, 
                q=1, 
                bounds=bounds, 
                num_restarts=num_restarts, 
                raw_samples=raw_samples)
            new_Y = torch.tensor([lookup[int(new_X[0][0]), int(new_X[0][1])]], 
                                 dtype=torch.float64, 
                                 device=device).reshape(-1, 1)

            #add new candidate
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

            #end timer and add
            toc = time.perf_counter() #end time
            _result.append([new_Y[0][0].item(), toc - tic, new_X[0]])


        #save all your queries
        torch.save(train_X, f"./output/{trait}/botorch{acq_name}_{kernel_name}_X_{seed}.npy")
        torch.save(train_Y, f"./output/{trait}/botorch{acq_name}_{kernel_name}_Y_{seed}.npy")

        #organize the list to have running best
        best = [0,0,0] # format is [time, best co-heritabilty]
        botorch_result = []
        for i in _result:
            if i[0] > best[0]:
                best = i
            botorch_result.append([best[0], i[1], best[2]]) # append [best so far, current time]
        print("Best From Run: ", best)

        #store results
        botorch_result = pd.DataFrame(botorch_result, columns=["Best", "Time", "Candidate"])
        botorch_result.to_csv(f"./output/{trait}/botorch{acq_name}_{kernel_name}_result_{seed}.npy", encoding='utf8') #store botorch search results

        #print full time
        toc = time.perf_counter() #end time
        print("BoTorch Took ", (toc-tic) ,"seconds")

main()


#add other acquisition function options
#make sure that data loading works correctly