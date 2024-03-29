### Downloading the data:
1. The coheritability data can be found in the [google drive](https://drive.google.com/drive/folders/1bdezexIEo9CdIViKiN_m_XlCUe3rJk7w).
2. Move files the data files from `./data/*.csv` on the Google Drive to `./test_functions` in the project folder.


## Running Bayesian Optimization (BO) Benchmarks
1. `cd Benchmarks`
2. Run `python main.py --env <trait> --n <number of iterations> --kernel <kernel> --acq <acquisition function>`
  Example: `python main.py --env narea --n 30 --kernel matern12 --acq EI`
   
