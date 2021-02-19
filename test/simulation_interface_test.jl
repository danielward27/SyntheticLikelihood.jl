using SyntheticLikelihood, Test

simulator = SyntheticLikelihood.test_simulator
summary = SyntheticLikelihood.test_summary
θ_vec = [1.0, 2, 3, 4, 5]
n_sim = 10

θ_array = [1.0 2; 3 4; 4 5]

x1 = simulate_n_s(θ_vec; simulator, summary, n_sim)
x2 = simulate_n_s(θ_array; simulator, summary)

@test size(x1) == (n_sim, length(θ_vec))
@test size(x2) == size(θ_array)  # Uses fact summary does nothing
