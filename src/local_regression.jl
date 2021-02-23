
"""
Quadratic transform. Bias term appended as first column internally.
Returns a tuple, with the matrix and the corresponding indices multiplied,
that give rise to each column. Note, indices [1, 1] corresponds to the bias
term (so indices compared to original matrix is shifted).
A bit naive so could be sped up but neat.

$(SIGNATURES)
"""
function quadratic_transform(X::AbstractMatrix)
    X = hcat(ones(size(X)[1]), X)  # Bias
    combinations = pairwise_combinations(size(X)[2])
    result = Matrix{Float64}(undef, size(X)[1], size(combinations)[1])
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
Struct that contains the information from the first local regression.
"""
struct Localμ
    μ::Float64
    ∂::Vector{Float64}
    ∂²::Matrix{Float64}
    ϵ::Vector{Float64}
end

"""
Gets the local behaviour of μ. Returns Localμ struct, containing the first and
second derivitive estimates as well as the regression residuals.

$(SIGNATURES)

# Arguments
- `θ_orig::AbstractVector` Original θ.
- `θ::AbstractMatrix` Peturbed θ (sampled from local area).
- `s::AbstractMatrix` Corresponding summary statistics to θ.
"""
function get_local_μ(;
    θ_orig::AbstractVector,
    θ::AbstractArray,
    s::AbstractArray)
    @assert size(θ)[1] == size(s)[1]

    # Center and carry out quadratic regression for each s
    θ = θ .- θ_orig'
    θ, combinations = quadratic_transform(θ)

    d = size(s)[2]
    μ = Vector{Localμ}(undef, d)

    for i in 1:d
       β, ŝ = linear_regression(θ, s[:, i])

       # Convert β to matrix
       n_features = floor(Int, 1/2 * (sqrt(8*length(β)+1) -1) + 0.1)
       β_mat = Matrix{Float64}(undef, n_features, n_features)

       for (i, row) in enumerate(eachrow(combinations))
           β_mat[row[1], row[2]] = β[i]  # Upper traingular
       end

       β_mat = Symmetric(β_mat)
       μ[i] = Localμ(β_mat[1,1], β_mat[2:end, 1],
                     β_mat[2:end, 2:end], ŝ-s[:, 1])
    end
    μ
end
