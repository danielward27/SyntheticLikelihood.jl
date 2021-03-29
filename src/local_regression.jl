
"""
Design matrix for quadratic regression. Bias term appended as first column
internally. Returns a tuple, with the matrix and the corresponding indices
multiplied, that give rise to each column. Note, indices [1, 1] corresponds to
the bias term (so indices compared to original matrix is shifted).

$(SIGNATURES)
"""
function quadratic_design_matrix(X::AbstractMatrix)
    X = [ones(size(X, 1)) X]  # Bias
    combinations = [(i,j) for i in 1:size(X, 2) for j in i:size(X, 2)]
    result = Matrix{Float64}(undef, size(X, 1), size(combinations, 1))

    # A bit naive so could be sped up but neat.
    for (i, idxs) in enumerate(combinations)
        result[:, i] = X[:, idxs[1]] .* X[:, idxs[2]]
    end
    result, combinations
end


"""
Carry out linear regression. X should have a bias column.
Returns tuple (β, ŷ).

$(SIGNATURES)
"""
function linear_regression(X::AbstractMatrix, y::AbstractVector)
    β = X \ y  # Linear regression
    ŷ = X * β
    (β = β, ŷ = ŷ)
end


"""
Struct that contains the estimated local properties of μ.

# Fields
- `μ::Float64` Means of the summary statistics.
- `∂::Vector{Float64}` First derivitives w.r.t. parameters (nₛ×n_θ).
- `∂²::Matrix{Float64}` Second derivitive w.r.t. parameters (nₛ×n_θ×n_θ).
- `ϵ::Vector{Float64}` Residuals of predicted summary statistics (nₛ×nₛᵢₘ).
"""
struct Localμ
    μ::Vector{Float64}
    ∂::Matrix{Float64}
    ∂²::Array{Float64, 3}
    ϵ::Matrix{Float64}
end

"""
Finds the local behaviour of the summary statistic mean μ.
Uses quadratic linear regression to approximate the mean, gradient and
hessian around `θᵢ`. Returns a `Localμ` struct (see above).

$(SIGNATURES)

# Arguments
- `θᵢ::AbstractVector` Original θ.
- `θ::AbstractMatrix` Peturbed θ (sampled from local area).
- `s::AbstractMatrix` Corresponding summary statistics to θ.
"""
function quadratic_local_μ(;
    θᵢ::AbstractVector,
    θ::AbstractMatrix,
    s::AbstractMatrix)
    @assert size(θ, 1) == size(s, 1)

    if s isa Vector
        nₛ = 1
    else
        nₛ = size(s, 2)
    end

    n_θ = size(θ, 2)
    nₛᵢₘ = size(θ, 1)

    μ = Vector{Float64}(undef, nₛ)
    ∂ = Matrix{Float64}(undef, nₛ, n_θ)
    ∂² = Array{Float64}(undef, nₛ, n_θ, n_θ)
    ϵ = Matrix{Float64}(undef, nₛᵢₘ, nₛ)

    # Center and carry out quadratic regression for each s
    θ = θ .- θᵢ'
    θ, combinations = quadratic_design_matrix(θ)

    for i in 1:nₛ
       β, ŝ = linear_regression(θ, s[:, i])

       # Convert β to matrix
       β_mat = Matrix{Float64}(undef, length(θᵢ) + 1, length(θᵢ) + 1)

       for (i, idxs) in enumerate(combinations)
           β_mat[idxs...] = β[i]  # Upper traingular
       end

       β_mat = Symmetric(β_mat)
       μ[i] = β_mat[1,1]
       ∂[i, :] = β_mat[2:end, 1]
       ∂²[i, :, :] = β_mat[2:end, 2:end]
       ϵ[:, i] = ŝ-s[:, i]
    end
    Localμ(μ, ∂, ∂², ϵ)
end


"""
Struct that contains the estimated local properties of Σ (the covariance matrix
of the summary statistics).

$(FIELDS)
- Σ The (nₛ×nₛ) (estimated) covariance matrix of the summary statistics.
- ∂ The (nₛ×nₛ×n_θ) matrix of estimated first derivitives of Σ.
"""
struct LocalΣ
    Σ::Symmetric{Float64}
    ∂::Array{Float64, 3}

    function LocalΣ(Σ, ∂)
        @assert size(Σ) == size(∂)[1:2]
        new(Σ, ∂)
    end
end


"""
Use a gamma distributed GLM with log link function to estimate the local properties
    of the covariance matrix of the statistics Σ.  θ should not have a bias term (added internally).

Specifically, this function:
- Creates a rough initial Σ estimate using `cov(ϵ)`.
- Estimates the diagonal elements Σⱼⱼ, and ∂Σⱼⱼ using local regression.
- Esimates off-diagonal elements of Σ by scaling the sample correlation matrix
    with √Σⱼⱼ (standard deviations).
- Esimate off-diagonal gradients ∂Σᵢⱼ by averaging the coefficients associated
with indices i and j.

$(SIGNATURES)

# Arguments
- `θᵢ::AbstractVector`  Original θ (used for centering).
- `θ::AbstractMatrix` Peturbed θ from local area.
- `ϵ::AbstractMatrix` Residuals from quadratic regression (n_sim × n_sumstats).
"""
function glm_local_Σ(;
    θᵢ::AbstractVector,
    θ::AbstractMatrix,
    ϵ::AbstractMatrix)

    nₛ = size(ϵ, 2)
    n_θ = length(θᵢ)
    θ = θ .- θᵢ'  # Center
    θ = hcat(ones(size(θ, 1)), θ)  # Bias
    ϵ² = ϵ.^2  # Distributed as ϵ² ∼ exp(ϕ + ∑vₖθₖ)z, z ∼ χ²(1)

    samp_Σ = cov(ϵ)
    samp_Ψ = Diagonal(samp_Σ)^-0.5 * samp_Σ * Diagonal(samp_Σ)^-0.5  # cor

    # Get coefficients of GLM
    coefs = Array{Float64}(undef, nₛ, n_θ+1)
    for i in 1:nₛ
        coefs[i, :] = coef(glm(θ, ϵ²[:, i], Gamma(), LogLink()))  # TODO: Add weights?
    end

    ϕ = coefs[:, 1]  # exp(ϕ) gives variance estimates
    v = coefs[:, 2:end]

    # Estimate the gradients
    ∂ = Array{Float64}(undef, nₛ, nₛ, n_θ)
    for i in 1:nₛ for j in 1:nₛ  # upper traingular
        ∂[i, j, :] = ∂[j, i, :] =
         samp_Ψ[i, j] * exp((ϕ[i] + ϕ[j])/2) .* ((v[i, :] .+ v[j, :])/2)
    end end

    # Estimate covariance matrix
    sds = Diagonal(exp.(ϕ/2))
    Σ = sds * samp_Ψ * sds

    LocalΣ(Symmetric(Σ), ∂)
end


"""
Estimate negative log-synthetic likelihood, and its gradient and hessian.
$(SIGNATURES)

## Arguments
- `θ` Starting parameter values.
- `P` Distribution used to peturb parameter value
    (e.g. 0 mean multivariate normal).
- `s_true` The observed summary statistics.
- `simulator` The simulator function taking parameter vector θ.
- `summary` The summary function taking output from the simulator.
- `n_sim` The number of peturbed points to use for the local regression.
- `eigval_threshold = 0.5` Minimum eigenvalue threshold for the estimated hessian.
    Negative eigenvalues are flipped and those smaller than the threshold are
    set to the threshold.
"""
function local_synthetic_likelihood(θ::Vector{Float64};
  s_true::Vector{Float64},
  simulator::Function,
  summary::Function=identity,
  P::Sampleable,
  n_sim::Integer,
  eigval_threshold::Float64 = 0.5
  )
  θᵢ = θ
  θ = peturb(θᵢ, P, n_sim)
  s = simulate_n_s(θ; simulator, summary)
  s = remove_invariant(s)
  μ = quadratic_local_μ(; θᵢ, θ, s)
  Σ = glm_local_Σ(; θᵢ, θ, μ.ϵ)
  l = local_synthetic_likelihood(μ, Σ, s_true; eigval_threshold)
  return l
end


# The objective gradient and hessian calculation after local regressions.
function local_synthetic_likelihood(
    μ::Localμ, Σ::LocalΣ,
    s_true::Vector{Float64};
    eigval_threshold::Float64)
  sᵒ = s_true
  n_θ = size(μ.∂, 2)
  ∂ = Vector{Float64}(undef, n_θ)
  ∂² = Matrix{Float64}(undef, n_θ, n_θ)

  qrΣ = qr(Σ.Σ)
  Σ⁻¹sᵒ₋μ = qrΣ \ (sᵒ - μ.μ)  # Precalculate Σ⁻¹(sᵒ-μ)

  # Gradient
  for k in 1:n_θ
    Σ⁻¹∂Σₖ = qrΣ \ Σ.∂[:, :, k]
    ∂[k] = (μ.∂[:,k]' * Σ⁻¹sᵒ₋μ +
        0.5*(sᵒ - μ.μ)' * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ - 0.5*tr(Σ⁻¹∂Σₖ))
  end

  # Hessian
  for k in 1:n_θ for l in k:n_θ  # Upper traingular of hessian matrix
    Σ⁻¹∂Σₗ = qrΣ \ Σ.∂[:, :, l]
    Σ⁻¹∂Σₖ = qrΣ \ Σ.∂[:, :, k]
    ∂²[k, l] = (
      μ.∂²[:, k, l]' * Σ⁻¹sᵒ₋μ - μ.∂[:, k]' * Σ⁻¹∂Σₗ * Σ⁻¹sᵒ₋μ - μ.∂[:, k]'* (qrΣ \ μ.∂[:, l])
      - μ.∂[:, l]' * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ - (sᵒ - μ.μ)' * Σ⁻¹∂Σₗ * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ
      + (1/2)*tr(Σ⁻¹∂Σₗ * Σ⁻¹∂Σₖ)
      )
  end end
  ∂² = Symmetric(∂²)
  neg_∂² = ensure_posdef(-∂², eigval_threshold)

  # Evaluate likelihood
  mvn = MvNormal(μ.μ, Σ.Σ)
  l = logpdf(mvn, sᵒ)

  return ObjGradHess(-l, -∂, neg_∂²)
end


## Bayesian

# For Bayesian analyses we also need the gradient and hessian of -prior:
function neg_prior_gradient(D, θ)
    f(θ) = -loglikelihood(D, θ)
    ForwardDiff.gradient(f, θ)
end

function neg_prior_hessian(D, θ)
    f(θ) = -loglikelihood(D, θ)
    ForwardDiff.hessian(f, θ)
end

"""
Estimate negative log-posterior, and its gradient and hessian. Uses a local
regressions to first estimate the gradient and hessian of the likelihood
function, using `local_synthetic_likelihood`, then uses the chain rule to
calculate the gradient of the posterior.

Prior gradients and hessians are calculated with
[DistributionsAD.jl](https://github.com/TuringLang/DistributionsAD.jl), so the
prior must be compatible with this package.

$(SIGNATURES)

## Arguments
- `θ` Starting parameter values.
- `prior` A vector of distributions (from Distributions package). Note that
    the distributions can be univariate or multivariate, but the overall dimension
    must match that of the θ (and the order must be consistent).
- `kwargs...` Key word arguments passed to `local_synthetic_likelihood`.
"""

function local_posterior(
    θ::Vector{Float64};
    prior::Vector{Sampleable},
    kwargs...
    )
    @assert length(θ) == sum([length(p) for p in prior])


    # TODO Handle cases where prior support is bounded?


    # Calculate likelihood
    l =  local_synthetic_likelihood(θ; kwargs...)




end
