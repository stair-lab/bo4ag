import torch
import gpytorch
from gpytorch.priors import HalfCauchyPrior, GammaPrior, Prior
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, SpectralMixtureKernel
from torch.nn import Module as TModule
import matplotlib.pyplot as plt


class SAASPrior(Prior):
    def __init__(self, num_dims, alpha=0.01):
        Prior.__init__(self)
        TModule.__init__(self)
        self.num_dims = num_dims
        self.alpha = alpha
        self._transform = None

    def sample(self, shape=torch.Size(), **kwargs):
        # Sample tau from a half-Cauchy
        tau_prior = HalfCauchyPrior(scale=self.alpha)
        tau_sample = tau_prior.rsample()

        # Sample rho for each dimension d
        rho_sample = torch.empty(self.num_dims)
        for d in range(self.num_dims):
            rho_prior = HalfCauchyPrior(scale=tau_sample)
            rho_sample[d] = rho_prior.sample()

        return rho_sample

    # monte carlo estimate of the saas prior mean
    def MCmean(self, num_samples=500):
        total = 0
        for _ in range(num_samples):
            total += self.sample()
        return total / num_samples


def getKernel(kernel_name, device):
    covar_module = None
    if kernel_name == None:
        return covar_module
    if kernel_name == "matern52":
        #covar_module = ScaleKernel(MaternKernel(nu=5 / 2, ard_num_dims=2,))
        covar_module = ScaleKernel(
            MaternKernel(
            nu=2.5,
            ard_num_dims=2,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
        outputscale_prior=GammaPrior(2.0, 0.15),)
    elif kernel_name == "matern32":
        covar_module = ScaleKernel(MaternKernel(nu=3 / 2, ard_num_dims=2,))
#     elif kernel_name == "matern12":
#         covar_module = ScaleKernel(MaternKernel(nu=1 / 2, ard_num_dims=2,))
#     elif kernel_name == "rbf":
#         covar_module = ScaleKernel(RBFKernel(ard_num_dims=2),)
#     elif kernel_name == "additive":
#         rbf = ScaleKernel(RBFKernel(ard_num_dims=2))
#         matern32 = ScaleKernel(MaternKernel(nu=3 / 2, ard_num_dims=2))
#         covar_module = ScaleKernel(rbf + matern32)
#     elif "spectral" in kernel_name:
#         _, num_mixtures = kernel_name.split("-")
#         num_mixtures = int(num_mixtures)
#         covar_module = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=2)
#     elif kernel_name == "saas":
#         pass
    else:
        print("Not a valid kernel")  # should also throw error
    
#     #set the prior here
#     lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
#     outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
#     if kernel_name in ["matern52", "matern32", "matern12", "rbf"]: #set to default priors
#         covar_module.base_kernel.lengthscale_prior = lengthscale_prior.to()
#         covar_module.outputscale_prior = outputscale_prior.to()
        
#     elif kernel_name == "saas":
#         # set the base kernel
#         covar_module = ScaleKernel(MaternKernel(nu=5 / 2, ard_num_dims=2))
#         # take a sample from saas prior
#         saas_prior = SAASPrior(num_dims=2)
#         length_scale = saas_prior.MCmean()  # use estimated mean as the length scale
#         # manually set the length scale
#         covar_module.base_kernel.lengthscale = length_scale
    return covar_module


def gp_mean_plot(model, file_name, device, title="Your Function", n=900):
    """
    Plots your (true or estimated) coheritability function.
    model: function that takes in two wavelength and outputs coh2 (estimate)
    n: n**2 is the number of points to plot the surface
    """
    # Generate data for the plot
    x1 = torch.linspace(0, 2150, n)
    x2 = torch.linspace(0, 2150, n)
    X1, X2 = torch.meshgrid(x1, x2)
    X = torch.stack([X1.reshape((-1)), X2.reshape((-1))]).T

    Z = model.posterior(X.to(device)).mean
    Z = Z.cpu().detach().numpy()

    # Create a contour plot
    plt.contourf(X1, X2, Z.reshape(n, n), cmap="viridis")
    plt.colorbar(label="Function Value")

    # plot top 1%
    # threshold = np.percentile(Z, 99)
    # threshold = torch.kthvalue(Z.flatten(), int(0.99 * n * n)).values
    # plt.contour(X1, X2, Z.reshape(n, n), levels=[threshold], colors='red', linewidths=2)

    # Add labels and title
    plt.xlabel("Wavelength 1")
    plt.ylabel("Wavelength 2")
    # plt.title(title)

    # show the plot
    # plt.show()

    # save the image here
    plt.savefig(file_name)
    return
