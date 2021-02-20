# Create noisy gaussian

# Toy example, noisy normal logpdf with means 1 and covariance 1
# Density plays role of summary statistic for local regression
# Provides simple sanity check between hessian and covariance matrix.
function noisy_normal_logpdf(θ::AbstractVector; μ::AbstractVector = [1,2,3])
    d = MvNormal(μ, 1)
    logpdf(d, θ) + rand()
end


"""
Quadratic transform. Returns Matrix. Bias term should be added prior to use.
Returns a tuple, with the matrix, and the indices multiplied.
A bit naive so could be sped up but neat.
"""
function quadratic_transform(X::AbstractMatrix)
    combinations = pairwise_combinations(size(X)[2])
    result = Matrix{Float64}(undef, size(X)[1], size(combinations)[1])

    for (i, idxs) in enumerate(eachrow(combinations))
        result[:, i] = X[:, idxs[1]] .* X[:, idxs[2]]
    end
    result, combinations
end

"""
Carry out quadratic regression. X should have a bias column.
Returns beta in a symetric matrix, where indices correspond to
combinations of variables (e.g. [1,1] would be intercept, [2:end, :]
is linear terms, the rest are quadratic terms).
"""
# TODO: Add test for quadratic regression (e.g. use deterministic case)

function quadratic_regression(X::AbstractMatrix, s::AbstractVector)
    n_features = size(X)[2]
    X, combinations = quadratic_transform(X)
    β = X \ s  # Linear regression

    β_mat = Matrix{Float64}(undef, n_features, n_features)

    for (i, row) in enumerate(eachrow(combinations))
        β_mat[row[1], row[2]] = β[i]
    end

    Symmetric(β_mat)
end


θ_mle = [1,2,3]
θ = peturb(θ_mle, [0.5, 0.5, 0.5], 100)
s = noisy_normal_logpdf.(eachrow(θ))

# Now have θ and s. Regress θ on s:
θ = θ .- θ_mle'  # Demean
θ = hcat(ones(size(θ)[1]), θ)
quadratic_regression(θ, s)



struct Localμ
    μ::Float64
    ∂μ::Vector{Float64}
    ∂μ²::Symmetric{Float64}
end

# outer constructor?



function get_∂μ(β)
    error("unimplemented")
end

function get_∂²μ(β, combinations)
    error("unimpimented")
end


# Use this knowledge? https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s?rq=1
