# Test posterior example. If this works then likelihood should too.
using SyntheticLikelihood, Test, Random, Distributions
seed = Random.seed!(1)

function simulator(θ::Vector{Float64})
  @assert length(θ) == 10
  d = MvNormal(θ, sqrt(0.1))
  rand(seed, d)
end

θ_true = zeros(10)
s_true = simulator(θ_true)

local_likelihood = LocalLikelihood(;
  simulator, s_true,
  init_P = MvNormal(fill(0.5, 10)),
  n_sim = 1000,
)

pULA = RiemannianULA(0.1)

init_θ = fill(5., 10)

data = run_sampler!(pULA, local_likelihood; init_θ, n_steps = 500)
θ = data[:θ][101:end, :] # Remove burn in


data = run_sampler!(pULA, local_posterior; init_θ, n_steps = 100)
θ = data[:θ][101:end, :] # Remove burn in

likelihood = MvNormal(θ_true, sqrt(0.1))
expected = SyntheticLikelihood.analytic_mvn_posterior(prior, likelihood)

@test isapprox(mean(expected), mean.(eachcol(θ)); atol = 1.5)
