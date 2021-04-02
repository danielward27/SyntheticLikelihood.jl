# Local regression
Local regressions can be used to estimate the gradient and hessian of the likelihood,
which can be used to improve the sampling efficieny of synthetic likelihood.

## Example
Here we consider a simple example, in which we infer the mean of a 10-dimensional multivariate normal distribution, using simulations from the distribution.

#### Define the simulator
The simulator must take a single positional argument, which is the parameter vector:

```@example 1
using SyntheticLikelihood, Distributions, Plots

# Define the simulator
function simulator(θ::Vector{Float64})
  @assert length(θ) == 10
  d = MvNormal(θ, sqrt(0.1))
  rand(d)
end

nothing # hide
```

#### Ground truth
The "true" parameters which we will aim to estimate, is just a vector of zeros. We can use this to generate a pseudo-observed data set `s_true`.

```@example 1
θ_true = zeros(10)
s_true = simulator(θ_true)
```

#### Defining how to estimate the likelihood
We can then define the hyperparameters for estimating the likelihood using local regression using [`LocalLikelihood`](@ref).

```@example 1
local_likelihood = LocalLikelihood(;
  simulator, s_true,
  P = MvNormal(fill(0.5, 10)),
  n_sim = 1000
)
nothing # hide
```

Note that if required a `summary` function can optionally be specified here, to summarise the output of the `simulator`.

#### Defining sampling method
We can then define how to sample from the distribution. Below I will use the [`PreconditionedLangevin`](@ref) sampler with a step size of 0.1.

```@example 1
plangevin = PreconditionedLangevin(0.1)
nothing # hide
```

#### Sampling
We can now define some initial parameter values, `init_θ`, and sample from the distribution using [`run_sampler!`](@ref):

```@example 1
init_θ = convert(Vector{Float64}, 1:10)
data = run_sampler!(plangevin, local_likelihood; init_θ, n_steps = 500)
nothing # hide
```

#### Plot the samples
```@example 1
param_names = reshape(["θ$i" for i in 1:10], (1,10))
plot(data.θ, label = param_names)
```

We can see that after the burn in period, samples are generally centered around the true parameter values (all zeros). More specifically, they are centered around `s_true` in this case, which are generally around zero.

## Bayesian inference
Given a prior, it is also simple to sample from the posterior instead of the likelihood. The prior should be specified using the distributions from [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/). A multivariate distribution can be used, or alternatively the prior can be formed from independent univariate priors using a `Product` distribution from Distributions.jl. For this example the prior is a multivariate normal centered around 5, with no correlation structure, and σ=0.5:

```@example 1
prior = MvNormal(fill(5, 10), 0.5)
nothing # hide
```

We can then define our objective using [`LocalPosterior`](@ref) and run the sampler again:

```@example 1
local_posterior = LocalPosterior(prior, local_likelihood)
data = run_sampler!(plangevin, local_posterior; init_θ, n_steps = 500)
plot(data.θ, label = param_names)
```

Note internally, this uses [`LocalLikelihood`](@ref) to estimate the the gradient and Hessian of the likelihood as before, and then uses automatic differentiation of the prior to get the gradient and Hessian of the prior. These can then be used to calculcate the gradient and Hessian of the posterior.

## Currently available "objectives"
```@docs
LocalLikelihood
LocalPosterior
```
