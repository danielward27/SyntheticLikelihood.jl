# Create noisy gaussian

# Toy example, noisy normal logpdf.
# Density plays role of summary statistic for local regression
# Provides simple sanity check between hessian and covariance matrix.
# d is dimension or summary statistics
function noisy_normal_logpdf(θ::AbstractVector, d, args)
    mvn = MvNormal(args...)
    fill(logpdf(mvn, θ), d) + rand(Normal(), d)
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
function linear_regression(X::AbstractMatrix, s::AbstractVector)
    β = X \ s  # Linear regression
    ŷ = X * β
    (β = β, ŷ = ŷ)
end


struct Localμ
    μ::Float64
    ∂μ::Vector{Float64}
    ∂μ²::Matrix{Float64}
    ϵ::Vector{Float64}
end

# Outer constructor for getting Localμ struct from
# quadratic regression coefficients.
function Localμ(
    β::AbstractVector,
    combinations::AbstractMatrix,
    ϵ::Vector{Float64})

    # Convert β to matrix
    n_features = floor(Int, 1/2 * (sqrt(8*length(β)+1) -1) + 0.1)
    β_mat = Matrix{Float64}(undef, n_features, n_features)

    for (i, row) in enumerate(eachrow(combinations))
        β_mat[row[1], row[2]] = β[i]  # Upper traingular
    end
    β_mat = Symmetric(β_mat)
    Localμ(β_mat[1,1], β_mat[2:end, 1], β_mat[2:end, 2:end], ϵ)
end


## Set up example problem

function simulator(
    θ::AbstractVector{Float64};
    Σ::AbstractMatrix{Float64})

    mvn = MvNormal(θ, Σ)
    rand(mvn, 1)
end

d = 4  # Dimension of summary
N = 100  # Number of peturbed θ
Σ_true = rand(d, d) |> X -> X'* X + I

s = simulator([1,2,3,4]; Σ = Σ_true)



θ_true = 1:5
p = length(θ_true)
P = fill(0.5, p)  # Diagonal vector to peturb θ
θ = peturb(θ_true, P, N)

Σ_true = rand(d, d) |> X -> X'* X + I
s = Array{Float64}(undef, N, d)

for i in 1:size(s)[1]
    s[i, :] = noisy_normal_logpdf(θ[i, :], [θ_true, Diagonal(P)])
end


noisy_normal_logpdf(θ[1, :], d, [θ_true, Σ_true])

# Now have θ and s. Center and regress θ on s:
θ = θ .- θ_true'
θ, combinations = quadratic_transform(θ)

# Loop through each summary statistic and get results
μ = Vector{Localμ}(undef, d)
for i in 1:d
    β, ŝ = linear_regression(θ, s[:, i])
    μ[i] = Localμ(β, combinations)
end

μ  # Vector of local behaviour of mu for each sum stat



# Use this knowledge? https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s?rq=1
