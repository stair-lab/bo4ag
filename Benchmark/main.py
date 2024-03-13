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
from gpytorch.kernels import MaternKernel, ScaleKernel


from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound #should maybe be qUpperConfidence Bound...
from botorch.acquisition import qKnowledgeGradient 
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qExpectedImprovement

#set device here
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 

def getLookup(trait):
    names = {"narea": "Narea", "sla": "SLA", "ps": "PLSR_SLA_Sorghum", "pn": "FS_PLSR_Narea"}
    trait = trait.lower()
    trait = names[trait]
    
    base_path = "/dfs/scratch0/ruhana/GenCor"
    path = f"{base_path}/Table-Env/table_env/envs/csv_files/{trait}_Full_Analysis.csv"
    csv_mat = np.genfromtxt(path, delimiter=',' )
    x_starter, x_end, y_starter, y_end, lookup = csv_mat[0,1], csv_mat[0,-1], csv_mat[1,0], csv_mat[-1,0], csv_mat[1:, 1:]
    return lookup

def getKernel(kernel_name):
    covar_module = None
    if kernel_name == "matern":
        covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=2)
            )     
    else:
        print("Not a valid kernel") #should also throw error
    return covar_module

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env', help='Environment to run search.')
    parser.add_argument('--kernel', default='matern',
                        help='Kernel function for the gaussian process')
    parser.add_argument('--acq', default='EI', help='Acquisition function')
    parser.add_argument('--n', default=300, help='Number of iterations')
    
    args = parser.parse_args()
    runBO(args)
    return 

def runBO(args):
    num_restarts = 128  
    raw_samples = 128
    n = args.n #replace this
    trait = args.env
    seeds = 5 #consider replacing this
    kernel = getKernel(args.kernel)
    
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
                best_f = 0/.1
                acq = UpperConfidenceBound(gp, beta=0.1)
            elif acq_name == "EI":
                acq = qExpectedImprovement(gp, best_f=max(train_Y))
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
        torch.save(train_X, f"./output/{trait}/botorch{acq_name}new_X_{seed}.npy")
        torch.save(train_Y, f"./output/{trait}/botorch{acq_name}new_Y_{seed}.npy")

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
        botorch_result.to_csv(f"./output/{trait}/botorch{acq_name}new_result_{seed}.npy", encoding='utf8') #store botorch search results

        #print full time
        toc = time.perf_counter() #end time
        print("BoTorch Took ", (toc-tic) ,"seconds")

main()


#add kernel option
#add other acquisition function options
#make sure that data loading works correctly