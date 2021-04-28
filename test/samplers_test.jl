# Test posterior example. If this works then likelihood should too.
using SyntheticLikelihood, Test, Random, Distributions
seed = Random.seed!(2)

n = 5

function simulator(θ::Vector{Float64})
  @assert lastindex(θ) == n
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

prior = Prior([MvNormal(fill(5,n), 0.5)])
local_posterior = LocalPosterior(; simulator, s_true, prior)
data = run_sampler!(rula, local_posterior; init_θ, n_steps = 700)
θ = data[:θ][301:end, :] # Remove burn in

likelihood = MvNormal(θ_true, sqrt(0.1))
expected = SyntheticLikelihood.analytic_mvn_posterior(prior.v[1], likelihood)

@test isapprox(mean(expected), mean.(eachcol(θ)); atol = 2)


# Test standard synthetic likelihood gives reasonable results
basic_posterior = BasicPosterior(; simulator, s_true, prior)
rwm = RWMetropolis(MvNormal(cov(expected)))

data = run_sampler!(rwm, basic_posterior; init_θ,
  n_steps = 4000, collect_data = [:θ, :accepted])

θ = data.θ[data.accepted, :]
θ = θ[100:end, :]

@test isapprox(mean(expected), mean.(eachcol(θ)); atol = 2)
