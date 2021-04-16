# Test posterior example. If this works then likelihood should too.
using SyntheticLikelihood, Test, Random, Distributions
seed = Random.seed!(2)

n = 5

function simulator(θ::Vector{Float64})
  @assert length(θ) == n
  d = MvNormal(θ, sqrt(0.1))
  rand(seed, d)
end

θ_true = zeros(n)
s_true = simulator(θ_true)

local_likelihood = LocalLikelihood(;
  simulator, s_true,
  P = MvNormal(fill(0.5, n)),
)

rula = RiemannianULA(0.2)
init_θ = fill(5., n)

data = run_sampler!(rula, local_likelihood; init_θ, n_steps = 500)
θ = data[:θ][101:end, :] # Remove burn in

@test isapprox(mean.(eachcol(θ)), zeros(n); atol = 2)

prior = MvNormal(fill(5,n), 0.2)
local_posterior = LocalPosterior(; simulator, s_true, prior)
data = run_sampler!(rula, local_posterior; init_θ, n_steps = 500)
θ = data[:θ][101:end, :] # Remove burn in

likelihood = MvNormal(θ_true, sqrt(0.1))
expected = SyntheticLikelihood.analytic_mvn_posterior(prior, likelihood)

@test isapprox(mean(expected), mean.(eachcol(θ)); atol = 2)
