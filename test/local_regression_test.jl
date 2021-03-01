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


## Second regression

# Diagonal values of Dᵢ=ϵᵢϵᵢᵀ matrix is just ϵ² for each sum stat
# Make residuals following true model ϵ² ∼ exp(ϕ + ∑vₖθₖ)z, z ∼ χ²(1)
function test_residuals(;ϕ, v, θ)
    ϵ² = Vector{Float64}(undef, size(θ, 1))
    for i in 1:size(θ, 1)
        ϵ²[i] = exp(ϕ .+ dot(v, θ[i, :]))*rand(Chisq(1))
    end
    ϵ²
end

θ_orig = [1., 2, 3]
θ = peturb(θ_orig, MvNormal(3, 1), 1000)
θ_centered = θ .- θ_orig'

# ϵ² model kwargs for two summary statistics
ϵ²_model_1 = (ϕ = 0.1, v = [0.1, 0.2, 0.3], θ = θ_centered)
ϵ²_model_2 = (ϕ = 0.5, v = [0.4, 0.5, 0.6], θ = θ_centered)

ϵ² = hcat(test_residuals(; ϵ²_model_1...),
          test_residuals(; ϵ²_model_2...))

Σ = glm_local_Σ(; θ_orig, θ, ϵ = sqrt.(ϵ²))

# GLM coefficients ≈ paramters of ϵ² model
@test isapprox(ϵ²_model_1.ϕ, log(Σ.Σ[1]); atol = 0.1)
@test isapprox(ϵ²_model_2.ϕ, log(Σ.Σ[2]); atol = 0.1)

@test isapprox(Σ.∂[1,1,:]./Σ.Σ[1], ϵ²_model_1.v; atol = 0.2)
@test isapprox(Σ.∂[2,2,:]./Σ.Σ[2], ϵ²_model_2.v; atol = 0.2)
