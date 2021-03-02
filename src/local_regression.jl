
"""
Design matrix for quadratic regression. Bias term appended as first column
internally. Returns a tuple, with the matrix and the corresponding indices
multiplied, that give rise to each column. Note, indices [1, 1] corresponds to
the bias term (so indices compared to original matrix is shifted).

$(SIGNATURES)
"""
function quadratic_design_matrix(X::AbstractMatrix)
    X = [ones(size(X, 1)) X]  # Bias
    combinations = pairwise_combinations(size(X, 2))
    result = Matrix{Float64}(undef, size(X, 1), size(combinations, 1))

    # A bit naive so could be sped up but neat.
    for (i, idxs) in enumerate(eachrow(combinations))
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
- `μ::Float64` Mean of the summary statistic.
- `∂::Vector{Float64}` First derivitive w.r.t. parameters.
- `∂²::Matrix{Float64}` Second derivitive w.r.t. parameters.
- `ϵ::Vector{Float64}` Residuals.
"""
struct Localμ
    μ::Float64
    ∂::Vector{Float64}
    ∂²::Matrix{Float64}
    ϵ::Vector{Float64}
end

"""
Finds the local behaviour of the summary statistic mean μ.
Uses quadratic linear regression to approximate the mean, gradient and
hessian around `θ_orig`. Returns a vector of `Localμ` structs (see above),
with length equal to the number of summary statistics.

$(SIGNATURES)

# Arguments
- `θ_orig::AbstractVector` Original θ.
- `θ::AbstractMatrix` Peturbed θ (sampled from local area).
- `s::AbstractMatrix` Corresponding summary statistics to θ.
"""
function quadratic_local_μ(;
    θ_orig::AbstractVector,
    θ::AbstractArray,
    s::AbstractArray)
    @assert size(θ, 1) == size(s, 1)

    # Center and carry out quadratic regression for each s
    θ = θ .- θ_orig'
    θ, combinations = quadratic_design_matrix(θ)

    if s isa Vector
        d = 1
    else
        d = size(s)[2]
    end

    μ = Vector{Localμ}(undef, d)
    for i in 1:d
       β, ŝ = linear_regression(θ, s[:, i])

       # Convert β to matrix
       β_mat = Matrix{Float64}(undef, length(θ_orig) + 1, length(θ_orig) + 1)

       for (i, row) in enumerate(eachrow(combinations))
           β_mat[row[1], row[2]] = β[i]  # Upper traingular
       end

       β_mat = Symmetric(β_mat)
       μ[i] = Localμ(β_mat[1,1], β_mat[2:end, 1],
                     β_mat[2:end, 2:end], ŝ-s[:, i])
    end
    μ
end


"""
Get an array of residuals from a vector of Localμ structs.
Returns (length of residuals × number of summary stats) matrix
"""
function get_residuals(μ_vec::Vector{Localμ})
    vecvec = [μ.ϵ for μ in μ_vec]
    reduce(hcat, vecvec)
end


"""
Struct that contains the estimated local properties of Σ (the covariance matrix
of the summary statistics).

# Fields
- Σ The (nₛ×nₛ) (estimated) covariance matrix of the summary statistics.
- ∂ The (nₛ×nₛ×n_θ) matrix of estimated first derivitives of Σ.
"""
struct LocalΣ
    Σ::Array{Float64, 2}
    ∂::Array{Float64, 3}

    function LocalΣ(Σ, ∂)
        @assert size(Σ) == size(∂)[1:2]
        new(Σ, ∂)
    end
end


"""
Use a gamma distributed GLM with log link function to estimate the local properties
    of Σ. θ should not have a bias term (added internally).

# Arguments
- `θ_orig::AbstractVector`  Original θ (used for centering).
- `θ::AbstractMatrix` Peturbed θ from local area.
- `ϵ::AbstractMatrix` Residuals from quadratic regression.
"""
function glm_local_Σ(;
    θ_orig::AbstractVector,
    θ::AbstractMatrix,
    ϵ::AbstractMatrix)

    n = size(ϵ, 2)
    N = length(θ_orig)
    θ = θ .- θ_orig'  # Center
    θ = hcat(ones(size(θ, 1)), θ)  # Bias
    ϵ² = ϵ.^2  # Distributed as ϵ² ∼ exp(ϕ + ∑vₖθₖ)z, z ∼ χ²(1)

    Σ = LocalΣ(cov(ϵ),  # Initialize with rough estimate
               Array{Float64}(undef, n, n, N))

    for j in 1:n
        coefs = coef(glm(θ, ϵ²[:, j], Gamma(), LogLink()))  # TODO: Add weights?
        ϕ = coefs[1]
        v = coefs[2:end]
        Σ.Σ[j, j] = exp(ϕ)
        Σ.∂[j, j, :] = v.*exp(ϕ)
    end
    Σ
end
