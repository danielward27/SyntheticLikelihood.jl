# Create noisy gaussian

# Toy example, noisy normal logpdf with means 1 and covariance 1
# Density plays role of summary statistic for local regression
# Provides simple sanity check between hessian and covariance matrix.

function noisy_normal_logpdf(θ::AbstractVector; μ::AbstractVector = [1,2,3])
    d = MvNormal(μ, 1)
    logpdf(d, θ) + rand()
end


"""
Quadratic transform. Bias term appended as first column internally.
Returns a tuple, with the matrix and the corresponding indices multiplied,
that give rise to each column. Note, indices [1, 1] corresponds to the bias
term (so indices compared to original matrix is shifted).
A bit naive so could be sped up but neat.
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
"""
# TODO: Add test for quadratic regression (e.g. use deterministic case)
# TODO also return y👒?
function linear_regression(X::AbstractMatrix, s::AbstractVector)
    β = X \ s  # Linear regression
    ŷ = X * β
    (β = β, ŷ = ŷ)
end


struct Localμ
    μ::Float64
    ∂μ::Vector{Float64}
    ∂μ²::Matrix{Float64}
end

# Outer constructor for getting Localμ struct from
# quadratic regression coefficients.
function Localμ(β::AbstractVector, combinations::AbstractMatrix)
    # Convert β to matrix
    n_features = floor(Int, 1/2 * (sqrt(8*length(β)+1) -1) + 0.1)
    β_mat = Matrix{Float64}(undef, n_features, n_features)

    for (i, row) in enumerate(eachrow(combinations))
        β_mat[row[1], row[2]] = β[i]  # Upper traingular
    end
    β_mat = Symmetric(β_mat)
    Localμ(β_mat[1,1], β_mat[2:end, 1], β_mat[2:end, 2:end])
end



θ_mle = [1,2,3]
θ = peturb(θ_mle, [0.5, 0.5, 0.5], 100)
s = noisy_normal_logpdf.(eachrow(θ))

# Now have θ and s. Center and regress θ on s:
θ = θ .- θ_mle'
θ, combinations = quadratic_transform(θ)
β, ŷ = linear_regression(θ, s)
μ = Localμ(β, combinations)

# Use this knowledge? https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s?rq=1
