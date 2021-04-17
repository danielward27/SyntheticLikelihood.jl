```@meta
EditURL = "<unknown>/ricker.jl"
```

# Ricker Model Example
To see how things work, its easiest to consider an example. Here we consider
the simulating from the noisily observed ricker map.

### Imports

```@example ricker
using SyntheticLikelihood, Distributions, StatsPlots, Random
Random.seed!(1)
nothing #hide
```

### Define the simulator
The simulator should take a vector of parameters.

```@example ricker
function ricker(r, ϕ, σ; init_n=5, n_iters=200, nburn=50)
  ϵ = randn(n_iters)*σ
  nₜ = init_n

  x = Vector{Int}(undef, n_iters)
  for i in 1:n_iters
    x[i]  = rand(Poisson(ϕ*nₜ))
    nₜ = r*nₜ*exp(-nₜ + ϵ[i])
  end
  x[nburn+1:n_iters]
end
```

Make parameter input a vector:

```@example ricker
ricker(θ::Vector{Float64}) = ricker(θ...)
nothing #hide
```

### Summary function
If no summary statistic function is used, then the summary defualts to the
identity. However, a summary function can be specified that summarises
the output of the simulator in to a vector.

```@example ricker
function ricker_summary(x)
    if all(x.==0)
      return [0., 0., length(x), 0]
    else
      s = [mean(x[x.>0]),
        median(x[x.>0]),
        sum(x.==0),
        sum(x.>10)]
      return s
    end
  end
nothing #hide
```

### Ground truth
As this is a toy example, we will generate "true" parameters, alongside a
"pseudo-observed" simulated dataset.

```@example ricker
θ_true = [6, 1, 0.6]
x_true = ricker(θ_true)
s_true = ricker_summary(x_true)
nothing #hide
```

### The prior
Priors can either be multivariate distribution from the Distributions.jl
package, or be specified as a `Product` distribution from the
Distributions.jl package (for independent priors for each parameter).
Below a `Product` distribution is used.

```@example ricker
prior = Product([Uniform(3, 10), Uniform(0, 5), Uniform(0, 5)])
nothing #hide
```

### `LocalPosterior`

The local regression MCMC technique estimates the
gradient and Hessian of the likelihood at each iteration. To achieve this
rather than carrying out many simulations at a single parameter value to
estimate the likelihood, many simulations from a "local" area around the
current θ value must be used. One can sample parameters consistent with
the data using [`LocalLikelihood`](@ref). However here we consider Bayesian
inference, so will use [`LocalPosterior`](@ref).

```@example ricker
local_posterior = LocalPosterior(;
  simulator = ricker,
  summary = ricker_summary,
  s_true,
  n_sim = 1000,
  prior,
)
nothing #hide
```

The above object describes how we want to estimate the posterior at each
step in the sampler. The other important parameter `P` is the initial proposal
distribution. This is what is used to generate `n_sim` peturbed parameters
to make the local regressions possible (we require variance in the parameters).
This parameter isn't shown as it is left to the default (inferred from the prior).
The proposal adapts based on the Hessian estimate at each iteration.

### The sampler
We can then sample from the posterior. Below I will use the Riemannian
Unadjusted Langevin sampler ([`RiemannianULA`](@ref)) with a step size of 0.1.
A simple explanation of this sampler is that it uses a Newton update,
and adds some noise at each iteration.

```@example ricker
rula = RiemannianULA(0.1)
init_θ = [8, 4, 0.1]
n_steps = 2000
data = run_sampler!(rula, local_posterior; init_θ, n_steps)
```

### Plotting the results
StatsPlots.jl provides most the tools required for plotting results.

```@example ricker
θ_names = ["r" "ϕ" "σ"]
plot(data.θ, layout = 3, xlabel = θ_names, labels = false)
```

We can remove the burn in and plot the marginal densities. The package
provides [`plot_prior_posterior_density`](@ref) to achieve this simply.

```@example ricker
samples = data.θ[1000:end, :]
plot_prior_posterior_density(
  prior, samples, θ_true; θ_names
)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

