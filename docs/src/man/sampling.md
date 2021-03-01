# Sampling
Samplers can either be initialised using a parameter vector θ, or using an AbstractSamplerState object. Below is an example to sample from a multivariate
normal density using the discretized langevin diffusion (Unadjusted Langevin Algorithm).

```@example 1
using SyntheticLikelihood, Distributions, Gadfly

d = MvNormal([10 5; 5 10])
init_θ = [-15., -15]
objective(θ) = logpdf(d, θ)
gradient(θ) = gradlogpdf(d, θ)

data, state = langevin_diffusion(init_θ;
        objective, gradient, step_size = 1,
        ξ = MvNormal([0.5 0; 0 0.5]), n_steps = 1000, collect_data = [:θ])

θ_samples = reduce(hcat, data[1])'
x = y = range(-20, 20; length=50)

contours = layer(z = (x, y) -> logpdf(d, [x,y]), x = x, y = y, Geom.contour)
points = layer(x = θ_samples[:,1], y = θ_samples[:,2], Geom.point,
                Theme(alphas=[0.2], default_color = "black"))

plot(contours, points)
```
