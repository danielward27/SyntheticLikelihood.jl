using SyntheticLikelihood, Test, Random, Distributions

Random.seed!(1)

# Sample from MVN
d = MvNormal([10 5; 5 10])
init_θ = [-15, -15]
objective(θ) = logpdf(d, θ)
gradient(θ) = gradlogpdf(d, θ)
n_steps = 1000

state = GradientState(init_θ, objective(init_θ), gradient(init_θ))

data, state = langevin_diffusion(;
        state, objective, gradient, step_size = 1,
        ξ = MvNormal([0.5 0; 0 0.5]), n_steps,
        data_to_collect = [:θ, :counter])

θ_samples = reduce(hcat, data[:θ])'
θ_samples = θ_samples[100:end, :]  # Remove burn in

@test isapprox(mean.(eachcol(θ_samples)), [0, 0]; atol = 2)
@test data[:counter] == Vector(1:n_steps)
