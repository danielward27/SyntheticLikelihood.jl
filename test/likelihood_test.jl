using SyntheticLikelihood, Test, Random

Random.seed!(1)

simulator = SyntheticLikelihood.test_simulator
summary = SyntheticLikelihood.test_summary
θ_true = [1.0, 2, 3, 4, 5]
n_sim = 100

s_true = simulator(θ_true) |> summary
l = synthetic_likelihood(θ_true; simulator, summary, s_true, n_sim)

# Peturbed true parameters should have lower likelihood
reps = 10
l_peturbed = Vector{Float64}(undef, reps)
for i in 1:reps
    θ = θ_true + randn(length(θ_true))*10
    l_peturbed[i] = synthetic_likelihood(θ; simulator, summary, s_true, n_sim)
end

@test all(l .> l_peturbed.-10)
