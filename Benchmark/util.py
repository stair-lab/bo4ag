from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, SpectralMixtureKernel

def getKernel(kernel_name):
    covar_module = None
    if kernel_name == None:
        return covar_module
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