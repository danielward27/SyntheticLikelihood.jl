using SyntheticLikelihood, Test, LinearAlgebra, Distributions, Random

Random.seed!(1)

X = [1 2; 4 3]
expected_result = [1 1 1 2 2 4; 1 4 16 3 12 9]
@test quadratic_design_matrix(X)[1] == expected_result

## Quadratic regression should be able to represent true quadratic perfectly
function quadratic(x1::Vector{Float64}, x2::Vector{Float64})
    x1 .+ 2x2 .+ 3x1.*x2 .+ 4x1.^2 .+ 5x2.^2
end

X = rand(10, 2)
y = quadratic(X[:, 1], X[:, 2])

X, combinations = quadratic_design_matrix(X)
β, ŷ = linear_regression(X, y)
@test isapprox(y, ŷ)


## Test quadratic_local_μ gives expected results for deterministic quadratic simulator

simulator = SyntheticLikelihood.deterministic_test_simulator

θ_orig = [2.0, 5]
P = MvNormal(length(θ_orig), 2)
θ = peturb(θ_orig, P, 100)
s = simulate_n_s(θ; simulator, summary=identity)

μ = quadratic_local_μ(;θ_orig, θ, s)

# Test means
@test isapprox(μ[1].μ, 2)
@test isapprox(μ[2].μ, 10)
@test isapprox(μ[3].μ, 25)

# Test first and second derivitives
@test isapprox(μ[1].∂[1], 1)
@test isapprox(μ[3].∂²[2,2], 1)
@test isapprox(μ[3].∂²[1,2], 0, atol = 1e-10)

# Residuals should all be zero as deterministic quadratic example
@test get_residuals(μ) == [μ[1].ϵ'; μ[2].ϵ'; μ[3].ϵ']'
@test isapprox(get_residuals(μ), fill(0., size(s)); atol = 1e-10)