# Samplers
To sample from a distribution, first a sampler object should be created.

The currently available samplers are shown below:

```@docs
Langevin
PreconditionedLangevin
```

The sampler object defines the hyperparameters of the sampler, and a function `local_approximation`, which takes `θ` and returns a `LocalApproximation` object, with fields `objective`, `gradient` and `hessian`. The gradient and hessian defualt to `nothing` if not required. This approach seems a bit convoluted (compared to e.g. seperately passing objective, gradient and Hessian functions), but it facilitates reusing calculations shared between calculating the objective, gradient and hessian, if desired.

The aim should be to explore around the minima of the function, so the objective could be the negative log-posterior, for example.

The sampler object can then passed to `run_sampler!`, to sample from the distribution:
```@docs
run_sampler!
```

Below is an example to sample from a multivariate normal density using the discretized langevin diffusion (Unadjusted Langevin Algorithm).

```@example 1
using SyntheticLikelihood, Distributions, Plots

# Sample from MVN
d = MvNormal([10 5; 5 10])
function local_approximation(θ)
    LocalApproximation(objective = -logpdf(d, θ),
                       gradient = -gradlogpdf(d, θ))
end

init_θ = [-15., -15]
n_steps = 1000

langevin = Langevin(;step_size = 1., local_approximation)
data = run_sampler!(langevin, init_θ, n_steps, [:θ, :counter])

θ_samples = data[:θ]

# Plot samples
x = y = range(-20, 20; length=50)
f(x, y) = -logpdf(d, [x,y])
X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map(f, X, Y)
p = contour(x, y, f)
scatter!(θ_samples[:,1], θ_samples[:,2], legend = false)
```


## Implementation details
To implement a new sampler, each sampler must:
- Be a subtype of `AbstractSampler`.
- Have a `get_init_state` method, which returns a SamplerState object given `init_θ` e.g. with signature `get_init_state(sampler::MySampler, init_θ::Vector{Float64})`.
- Have an `update!` method, taking the sampler and the `SamplerState` object, e.g. `update!(sampler::MySampler, state::SamplerState)`. This updates the state (parameters, gradients etc, and sampler object if applicable).
