
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
hessian around `θ_orig`. Returns a `Localμ` struct (see above).

$(SIGNATURES)

# Arguments
- `θ_orig::AbstractVector` Original θ.
- `θ::AbstractMatrix` Peturbed θ (sampled from local area).
- `s::AbstractMatrix` Corresponding summary statistics to θ.
"""
function quadratic_local_μ(;
    θ_orig::AbstractVector,
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
    θ = θ .- θ_orig'
    θ, combinations = quadratic_design_matrix(θ)

    for i in 1:nₛ
       β, ŝ = linear_regression(θ, s[:, i])

       # Convert β to matrix
       β_mat = Matrix{Float64}(undef, length(θ_orig) + 1, length(θ_orig) + 1)

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
- `θ_orig::AbstractVector`  Original θ (used for centering).
- `θ::AbstractMatrix` Peturbed θ from local area.
- `ϵ::AbstractMatrix` Residuals from quadratic regression (n_sim × n_sumstats).
"""
function glm_local_Σ(;
    θ_orig::AbstractVector,
    θ::AbstractMatrix,
    ϵ::AbstractMatrix)

    nₛ = size(ϵ, 2)
    n_θ = length(θ_orig)
    θ = θ .- θ_orig'  # Center
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
Log-synthetic likelihood, and its gradient and hessian.

$(FIELDS)
"""
struct LocalSyntheticLikelihood
    l::Float64
    ∂::Vector{Float64}
    ∂²::Matrix{Float64}
end

function LocalSyntheticLikelihood(μ::Localμ, Σ::LocalΣ, sᵒ::Vector{Float64})
  n_θ = size(μ.∂, 2)
  ∂ = Vector{Float64}(undef, n_θ)
  ∂² = Matrix{Float64}(undef, n_θ, n_θ)

  # TODO: Drop low variance summary statistics

  qrΣ = qr(Σ.Σ)
  Σ⁻¹sᵒ₋μ = qrΣ \ (sᵒ - μ.μ)  # Precalculate Σ⁻¹(sᵒ-μ)

  # Gradient
  for k in 1:n_θ
    Σ⁻¹∂Σₖ = qrΣ \ Σ.∂[:, :, k]
    ∂[k] = μ.∂[:,k]' * Σ⁻¹sᵒ₋μ +
      0.5*(sᵒ - μ.μ)' * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ - 0.5*tr(Σ⁻¹∂Σₖ)
  end

  # Hessian
  for k in 1:n_θ for l in k:n_θ  # Upper traingular of hessian matrix
    Σ⁻¹∂Σₗ = qrΣ \ Σ.∂[:, :, l]
    Σ⁻¹∂Σₖ = qrΣ \ Σ.∂[:, :, k]
    ∂²[k, l] =
      μ.∂²[:, k, l]' * Σ⁻¹sᵒ₋μ - μ.∂[:, k]' * Σ⁻¹∂Σₗ * Σ⁻¹sᵒ₋μ - μ.∂[:, k]'* (qrΣ \ μ.∂[:, l])
      - μ.∂[:, l]' * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ - (sᵒ - μ.μ)' * Σ⁻¹∂Σₗ * Σ⁻¹sᵒ₋μ
      + (1/2)*tr(Σ⁻¹∂Σₗ * Σ⁻¹∂Σₖ)
  end end
  ∂² = Symmetric(∂²)

  # Evaluate likelihood
  mvn = MvNormal(μ.μ, Σ.Σ)
  l = logpdf(mvn, sᵒ)

  return LocalSyntheticLikelihood(l, ∂, ∂²)
end


function LocalSyntheticLikelihood(;
  θ_orig::Vector{Float64}, s_true::Vector{Float64},
  simulator::Function, summary::Function=identity,
  P::Sampleable, n_sim::Integer
  )

  θ = peturb(θ_orig, P, n_sim)
  s = simulate_n_s(θ; simulator, summary)
  μ = quadratic_local_μ(; θ_orig, θ, s)
  Σ = glm_local_Σ(; θ_orig, θ, μ.ϵ)
  l = LocalSyntheticLikelihood(μ, Σ, s_true)
  return l
end
