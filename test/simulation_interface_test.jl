using SyntheticLikelihood, Test, Distributions

simulator = SyntheticLikelihood.deterministic_test_simulator
θ_true = [2., 5]
n_sim = 100

d = MvNormal(length(θ_true), 2)
θ = peturb(θ_true, d, n_sim)

s1 = simulate_n_s(θ_true; simulator, summary=identity, n_sim=10)
s2 = simulate_n_s(θ; simulator, summary=identity)

@test isapprox(s1, reduce(hcat, fill([2, 10, 25], 10))')
@test isapprox(s2[:,1], θ[:,1])  # Would fail if data race occurs