# Sampling
Samplers can either be initialised using an initial parameter vector θ, or using an AbstractSamplerState object. The latter is returned at the end of a sampling run, and provides a convenient way to continue sampling further.

Below is an example to sample from a multivariate
normal density using the discretized langevin diffusion (Unadjusted Langevin Algorithm).

```@example 1
using SyntheticLikelihood, Distributions, Gadfly

d = MvNormal([10 5; 5 10])
init_θ = [-15., -15]
objective(θ) = -logpdf(d, θ)
gradient(θ) = -gradlogpdf(d, θ)

data, state = langevin_diffusion(
        init_θ; objective, gradient, step_size = [1.,1],
        n_steps = 1000, collect_data = [:θ])

θ_samples = data[:θ]
x = y = range(-20, 20; length=50)

contours = layer(z = (x, y) -> logpdf(d, [x,y]), x = x, y = y, Geom.contour)
points = layer(x = θ_samples[:,1], y = θ_samples[:,2], Geom.point,
                Theme(alphas=[0.2], default_color = "black"))

plot(contours, points)
```
