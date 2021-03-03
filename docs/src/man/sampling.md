# Sampling
To sample from a distribution, first a sampler object should be created. This defines the hyperparameters of the sampler, the objective function and the gradient/hessian if appropriate. When passing a function (objective/gradient/hessian) to a sampler, it should take a parameter vector θ as its first and only required argument. The aim should be to explore around the minima of the function, so the objective could be the negative log-posterior, for example.

The sampler object can then passed to `run_sampler!`, to sample from the distribution.


Currently available samplers are:
- `Langevin` which uses the update θ = θ .- η ./ 2 .* ∇ .+ ξ
    where η is the step size, ∇ the gradient, and ξ is `MvNormal(η)`.
- `PreconditionedLangevin` which uses the update θ = θ - H⁻¹ * η/2 * ∇ .+ ξ

Below is an example to sample from a multivariate normal density using the discretized langevin diffusion (Unadjusted Langevin Algorithm).

```@example 1
using SyntheticLikelihood, Distributions, Gadfly

# Sample from MVN
d = MvNormal([10 5; 5 10])
objective(θ) = -logpdf(d, θ)
gradient(θ) = -gradlogpdf(d, θ)
init_θ = [-15., -15]
n_steps = 1000

langevin = Langevin(;step_size = 1., objective, gradient)
data = run_sampler!(langevin, init_θ, n_steps, [:θ, :counter])

θ_samples = data[:θ]
x = y = range(-20, 20; length=50)

contours = layer(z = (x, y) -> logpdf(d, [x,y]), x = x, y = y, Geom.contour)
points = layer(x = θ_samples[:,1], y = θ_samples[:,2], Geom.point,
                Theme(alphas=[0.2], default_color = "black"))

plot(contours, points)
```


## Implementation details
To implement a new sampler, each sampler must:
- Be a subtype of `AbstractSampler`.
- Have a `get_init_state` method, which returns a SamplerState object given `init_θ` e.g. with signature `get_init_state(sampler::MySampler, init_θ::Vector{Float64})`.
- Have an `update!` method, taking the sampler and the `SamplerState` object `update!(sampler::MySampler, state::SamplerState)`.

- Each sampler must be a subtype of `AbstractSampler`.


have an the following methods`update!` method, where the
function update!(sampler::Langevin, state::SamplerState)
