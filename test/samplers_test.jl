using SyntheticLikelihood, Test, Random, Distributions

Random.seed!(1)

# Sample from MVN
d = MvNormal([10 5; 5 10])
objective(θ) = -logpdf(d, θ)
gradient(θ) = -gradlogpdf(d, θ)
init_θ = [-15., -15]
n_steps = 1000

langevin = Langevin(;step_size = [1., 1], objective, gradient)
data = run_sampler!(langevin, init_θ, n_steps, [:θ, :counter])

θ = data[:θ][100:end, :] # Remove burn in
@test isapprox(mean.(eachcol(θ)), [0, 0]; atol = 2)
@test data[:counter] == Vector(1:n_steps)
