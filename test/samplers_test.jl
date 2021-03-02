using SyntheticLikelihood, Test, Random, Distributions

Random.seed!(1)

# Sample from MVN
d = MvNormal([10 5; 5 10])
init_θ = [-15., -15]
objective(θ) = -logpdf(d, θ)
gradient(θ) = -gradlogpdf(d, θ)
n_steps = 1000

data, state = langevin_diffusion(
        init_θ; objective, gradient, step_size = [1., 1],
        n_steps, collect_data = [:θ, :counter])

θ_samples = data[:θ][100:end, :] # Remove burn in

@test isapprox(mean.(eachcol(θ_samples)), [0, 0]; atol = 2)
@test data[:counter] == Vector(1:n_steps)
