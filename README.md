## Installation 
1. `git clone https://github.com/stair-lab/bo4ag`
2. `cd your_repository`
3. Create a `python 3.10` conda environment 
4. `pip install -r requirements.txt`


### Downloading the data:
1. The coheritability data can be found on [huggingface](https://huggingface.co/datasets/stair-lab/coh2 or [google drive](https://drive.google.com/drive/folders/1bdezexIEo9CdIViKiN_m_XlCUe3rJk7w).
3. Download the files to `./Benchmark/data`.

## Running Bayesian Optimization (BO) Benchmarks
1. `cd Benchmarks`
2. Run `python run_BO.py --env <environment>`

**Options:**
| Argument      | Description                                            | Default Value | Example            |
|---------------|--------------------------------------------------------|---------------|--------------------|
| `--env`       | Environment to run the search (e.g. narea, sla, pn, ps)                          | None          | `--env narea` |
| `--kernel`    | Kernel function for the Gaussian process              | `rbf`         | `--kernel matern52` |
| `--acq`       | Acquisition function                                   | `EI`          | `--acq EI`         |
| `--transform` | Transforming on the search space                       | None          | `-- transform log` |
| `--n`         | Number of iterations                                   | `300`         | `--n 300`          |
| `--gpu`       | GPU id to run the job                                  | `0`           | `--gpu 0`          |
| `--run_name`  | Name of the folder to move outputs                     | None          | `--run_name test` |

You can copy and paste this markdown table into your README.md file.

**Example:** `python main.py --env narea --n 30 --kernel matern12 --acq EI --run_name test`
   
## Fitting A Surrogate Model
1. `cd Benchmarks`
2. Run `python gp_fit.py --env <trait> --n <number of iterations> --kernel <kernel> --acq <acquisition function>`
