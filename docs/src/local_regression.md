# Local regression
Local regressions can be used to estimate the gradient and hessian of the likelihood using [`local_synthetic_likelihood`](@ref).

Below is an example, inferring the means of a 10 dimensional multivariate normal distribution (with constant covariance), using a single observed sample from the distribution.

```@example 1
using SyntheticLikelihood, Distributions, Plots

# Define the simulator
function simulator(θ::Vector{Float64})
  @assert length(θ) == 10
  d = MvNormal(θ, sqrt(0.1))
  rand(d)
end

# Generate pseudo-observed data
θ_true = zeros(10)
s_true = simulator(θ_true)

```

We can then define how to sample from the distribution. Below I will use the [`PreconditionedLangevin`](@ref) sampler with a step size of 0.1, and use [`local_synthetic_likelihood`](@ref) to estimate the gradient and Hessian of the likelihood. The hyperparameters to be used with [`local_synthetic_likelihood`](@ref) are passed as key word arguments to the sampler:

```@example 1
P = MvNormal(fill(0.5, 10))  # Local area in parameter space
n_sim = 1000  # Simulations used at each iteration
θ_orig = convert(Vector{Float64}, 1:10)  # Starting parameter values.

pl = PreconditionedLangevin(
  0.1, local_synthetic_likelihood; s_true, simulator, P, n_sim
  )
```

Now we can cary out the sampling and plot the results.
```@example 1
data = run_sampler!(pl, θ_orig, 1000)

# Plot the samples
param_names = reshape(["θ$i" for i in 1:10], (1,10))
plot(data.θ, label = param_names)
```

Plotting the marginals after removing the burn in period:
```@example 1
histogram(data.θ[250:end, :], label = param_names, layout = (2,5), size = (800, 600))
```
We can see the samples are generally centered around the true parameter values (all zeros). More specifically, they are centered around `s_true` in this case, which are generally around zero.


```@docs
local_synthetic_likelihood
```
