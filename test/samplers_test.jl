using SyntheticLikelihood, Test, Random, Distributions
Random.seed!(1)

# Sample from MVN
function local_approximation(θ)
    d = MvNormal([10 5; 5 10])
    LocalApproximation(objective = -logpdf(d, θ),
                       gradient = -gradlogpdf(d, θ))
end

init_θ = [-15., -15]
n_steps = 1000

langevin = Langevin(1., local_approximation)
data = run_sampler!(langevin, init_θ, n_steps, [:θ, :counter])

θ = data[:θ][101:end, :] # Remove burn in
@test isapprox(mean.(eachcol(θ)), [0, 0]; atol = 2)
@test data[:counter] == Vector(1:n_steps)
