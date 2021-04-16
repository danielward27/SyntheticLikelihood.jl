using SyntheticLikelihood, Test, Distributions, Random, LinearAlgebra, ForwardDiff

quadratic_design_matrix = SyntheticLikelihood.quadratic_design_matrix
linear_regression = SyntheticLikelihood.linear_regression
simulator = SyntheticLikelihood.deterministic_test_simulator

Random.seed!(1)

X = [1 2; 4 3]

expected_result = [1 1 2 1 2 4; 1 4 3 16 12 9]
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


θᵢ = [2.0, 5]
P = MvNormal(length(θᵢ), 2)
θ = peturb(θᵢ, P; n = 100)
s = simulate_n_s(θ; simulator, summary=identity)
μ = quadratic_local_μ(;θᵢ, θ, s)

# Test means
@test isapprox(μ.μ, [2,10,25])

# Test first and second derivitives
@test isapprox(μ.∂[1, 1], 1)
@test isapprox(μ.∂²[3,2,2], 1)
@test isapprox(μ.∂²[3, 1,2], 0, atol = 1e-10)

# Residuals should all be zero as deterministic quadratic example
@test isapprox(μ.ϵ, fill(0., size(s)); atol = 1e-10)


# Check regression works with summary vector length 1
s = s[:, 1]
s = reshape(s, 100, 1)
μ = quadratic_local_μ(;θᵢ, θ, s)
@test isapprox(μ.μ[1], 2)
@test size(μ.∂²) == (1,2,2)

## Second regression
@test_throws AssertionError LocalΣ(fill(1, 3, 3), fill(1, 3, 2, 10))

# Diagonal values of Dᵢ=ϵᵢϵᵢᵀ matrix is just ϵ² for each sum stat
# Make residuals following true model ϵ² ∼ exp(ϕ + ∑vₖθₖ)z, z ∼ χ²(1)
function test_residuals(;ϕ, v, θ)
    ϵ² = Vector{Float64}(undef, size(θ, 1))
    for i in 1:size(θ, 1)
        ϵ²[i] = exp(ϕ .+ dot(v, θ[i, :]))*rand(Chisq(1))
    end
    ϵ²
end

θᵢ = [1., 2, 3]
θ = peturb(θᵢ, MvNormal(3, 1); n = 1000)
θ_centered = θ .- θᵢ'

# ϵ² model kwargs for two summary statistics
ϵ²_model_1 = (ϕ = 0.1, v = [0.1, 0.2, 0.3], θ = θ_centered)
ϵ²_model_2 = (ϕ = 0.5, v = [0.4, 0.5, 0.6], θ = θ_centered)

ϵ² = [test_residuals(; ϵ²_model_1...) test_residuals(; ϵ²_model_2...)]


Σ = glm_local_Σ(; θᵢ, θ, ϵ = sqrt.(ϵ²))


# GLM coefficients ≈ paramters of ϵ² model
Σⱼⱼ = diag(Σ.Σ)
@test isapprox(ϵ²_model_1.ϕ, log(Σⱼⱼ[1]); atol = 0.1)
@test isapprox(ϵ²_model_2.ϕ, log(Σⱼⱼ[2]); atol = 0.1)

@test isapprox(Σ.∂[1,1,:]./Σⱼⱼ[1], ϵ²_model_1.v; atol = 0.1)
@test isapprox(Σ.∂[2,2,:]./Σⱼⱼ[2], ϵ²_model_2.v; atol = 0.1)



## Test glm_local_Σ using known expected values and gradients

# Make artificial example
seed = MersenneTwister(1)
∂ = Array{Float64}(undef, 3, 3, 2)

function rand_pd(seed)
        A = randn(seed, 3,3); A = A'*A; A = (A + A')/2
end

∂[:, :, 1] = rand_pd(seed)
∂[:, :, 2] = rand_pd(seed)

true_Σ = LocalΣ(Symmetric(rand_pd(seed)+ 100I), ∂)

# Simulate residuals under this model
θ = [randn(seed, 100000) randn(seed, 100000)]

ϵ = Array{Float64}(undef, size(θ, 1), size(true_Σ.Σ, 1))
for i in 1:size(ϵ, 1)
        model_Σ = true_Σ.Σ + true_Σ.∂[:, :, 1].*θ[i, 1] + true_Σ.∂[:, :, 2].*θ[i, 2]
        ϵ[i, :] = rand(seed, MvNormal(model_Σ))
end

estimted_Σ = glm_local_Σ(; θᵢ = zeros(size(θ, 2)), θ, ϵ)
sample_Σ = cov(ϵ)

# Compare sample covariance to estimated covariance
norm(sample_Σ - true_Σ.Σ)
norm(estimted_Σ.Σ - true_Σ.Σ)

# Are the diagonal elements improved?
norm(diag(sample_Σ) - diag(true_Σ.Σ))
norm(diag(estimted_Σ.Σ) - diag(true_Σ.Σ))
# Sometimes does worse, sometimes does better.

# Check gradient estimates are improved compared to assuming 0
@test norm(estimted_Σ.∂ - true_Σ.∂) < norm(true_Σ.∂)


## Test automatic differentiation of priors (for product and mv dists)

log_prior_gradient = SyntheticLikelihood.log_prior_gradient
log_prior_hessian = SyntheticLikelihood.log_prior_hessian

sd = 2.
prod_dist = Product([Normal(1,sd), Normal(2,sd), Normal(3,sd)])
mv_dist = MvNormal([1,2,3], sd)

θ = [1.,2,3]


@test log_prior_gradient(prod_dist, θ) ≈ [0,0,0]
@test log_prior_gradient(mv_dist, θ) ≈ [0,0,0]

@test log_prior_hessian(mv_dist, θ) ≈ -Diagonal(fill(1/sd^2, 3))
@test log_prior_hessian(prod_dist, θ) ≈ -Diagonal(fill(1/sd^2, 3))


## test posterior_ogh matches

test_θ = rand(10)
prior = MvNormal(rand(10), Diagonal(rand(10)))
likelihood = MvNormal(rand(10), Diagonal(rand(10)))

# Use product of two normals to check calculation correct

expected = begin
    posterior = SyntheticLikelihood.analytic_mvn_posterior(prior, likelihood)
    obj = loglikelihood(posterior, test_θ)
    expected_∇ = gradlogpdf(posterior, test_θ)
    expected_H = ForwardDiff.hessian(θ -> loglikelihood(posterior, θ), test_θ)
    ObjGradHess(-obj, -expected_∇, -Symmetric(expected_H))
end



f(θ) = loglikelihood(likelihood, θ)
neg_likelihood_ogh = ObjGradHess(
    -f(test_θ),
    -ForwardDiff.gradient(f, test_θ),
    Symmetric(-ForwardDiff.hessian(f, test_θ))
)


actual = SyntheticLikelihood.posterior_calc(
    prior, neg_likelihood_ogh, test_θ
    )

@test actual.objective != expected.objective  # Proportional
@test actual.gradient ≈ expected.gradient  # Independent of p(x)
@test actual.hessian ≈ expected.hessian  # Independent of p(x)
