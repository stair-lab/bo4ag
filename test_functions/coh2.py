import pandas as pd
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from typing import Optional


class Coh2(BaseTestProblem):
    def __init__(
        self,
        type,
	base_path: str = ".",
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = 2
        self._bounds = [
            [350, 2500], 
            [350, 2500], 
        ]
        self.num_objectives = 1

        #set lookup table path based of type here:
        #base_path = "."
        self.lookup = pd.read_csv(f"{base_path}/test_functions/{type}_coh2.csv")
        self.lookup = torch.tensor(self.lookup.values, dtype=torch.float64)
        
        super().__init__(
                negate=negate, noise_std=noise_std)

    def cfun(self, x):
        #print(x)
        x1 = int(x[0]) - 350 #shift over for indexing and round to int
        x2 = int(x[1]) - 350
        
        if self.lookup[x1][x2].detach().isnan().any():
            i = 0
            x1_new = x1
            x2_new = x2
            while self.lookup[x1_new][x2_new].detach().isnan().any():
                if i % 4 == 0 and x1 + (i//4) <= 2150:
                    x1_new = x1 + (i//4)
                elif i % 4 == 1 and x1 - (i//4) >= 0:
                    x1_new = x1 - (i//4) 
                elif i % 4 == 2 and x2 + (i//4) <= 2150:
                    x2_new = x2 + (i//4)
                elif i % 4 == 3 and x2 - (i//4) >= 0:
                    x2_new = x2 - (i//4)
                i = i + 1
            return self.lookup[x1_new][x2_new].detach() 
            
        return self.lookup[x1][x2].detach()

    def c_batched(self, X):
        return torch.stack([self.cfun(x) for x in X]).to(X)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.c_batched(X).to(X)



#func = Coh2("narea", base_path="/lfs/turing2/0/ruhana/bnn-bo/")
#import torch 
#x = torch.tensor([[350,350]], dtype=torch.float64)
#print(func(x))
