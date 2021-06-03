# # Example
# To demonstrate how the package works, we can consider a toy example:
# - We have observed a single sample from a multivariate normal distribution.
# - We have a simulator which takes mean vector θ and samples from a multivariate normal distribution.
# - We will aim to estimate the mean vector θ using simulations.

# ### Imports
using SyntheticLikelihood, Distributions, StatsPlots
using Random #hide
Random.seed!(1) #hide
nothing #hide

# ### Define the simulator
# The simulator should take a vector of parameters. Here, the simulator
# samples from a 3-d multivariate normal with mean θ and diagonal covariance
# with standard deviation 0.1.
function simulator(θ)
  sds = fill(√0.1, 3)
  d = MvNormal(θ, sds)
  rand(d)
end

sds = fill(√0.1, 3)
MvNormal(fill(0,3), sds)

# ### Ground truth
# As this is a toy example, we will generate ground truth parameters, alongside a
# "pseudo-observed" simulated dataset.
θ_true = fill(1, 3)
s_true = simulator(θ_true)
nothing #hide

# ### Define the prior
# The prior is specified as a vector of univariate and/or multivariate 
# distributions. Here we use a multivariate normal prior.
# The distributions should be from [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/).
prior = Prior([MvNormal(fill(2., 3))])

# ### `LocalPosterior`
# We can then define the hyperparameters associated with using local regressions
# to estmate the log-likelihood (and subsequently log-posteior) gradient and Hessian.
# Here we will use 1000 simulations at each sampler step.
local_posterior = LocalPosterior(;
  simulator,
  s_true,
  n_sim = 1000,
  prior,
)
nothing #hide

# ### The sampler
# We can then sample from the posterior. Below I will use the Riemannian
# Unadjusted Langevin sampler ([`RiemannianULA`](@ref)) with a step size of 0.5.
# This results in a tuple containing the samples along with other information.
rula = RiemannianULA(0.5)
init_θ = fill(0., 3)
n_steps = 1000
results = run_sampler!(rula, local_posterior; init_θ, n_steps, progress = false)

# ### Plotting the results
# We can plot a corner plot to visualise the posterior using StatsPlots
cornerplot(results.θ[200:end, :], label = ["θ$i" for i in 1:3], markercolor = :plasma)
