"""
Struct that contains the estimated local properties of μ (the expected values
of the summary statistics).

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
