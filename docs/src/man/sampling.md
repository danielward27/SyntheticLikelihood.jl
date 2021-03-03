# Sampling
Samplers can either be initialised using an initial parameter vector θ, or using an AbstractSamplerState object. The latter is returned at the end of a sampling run, and provides a convenient way to continue sampling further.

Below is an example to sample from a multivariate normal density using the discretized langevin diffusion (Unadjusted Langevin Algorithm).

```@example 1
using SyntheticLikelihood, Distributions, Gadfly

# Sample from MVN
d = MvNormal([10 5; 5 10])
objective(θ) = -logpdf(d, θ)
gradient(θ) = -gradlogpdf(d, θ)
init_θ = [-15., -15]
n_steps = 1000

langevin = Langevin(;step_size = [1., 1], objective, gradient)
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
- Each sampler must be a subtype of `AbstractSampler`.


have an the following methods`update!` method, where the
function update!(sampler::Langevin, state::SamplerState)
