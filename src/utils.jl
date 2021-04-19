
"""
Peturb a vector using a user specified distribution (often MVN zero mean).
Returns array of size (n, length(θ)). If  a prior is provided, proposals
are checked to have prior support using insupport and are resampled if not.

$(SIGNATURES)

# Arguments
- `θ` Parameter to peturb.
- `d` Distribution from which to sample (see Distributions.jl).
- `n = 1` Number of peturbed vectors to return.
- `valid_params` Return true if θ vector is valid, and false if invalid.
"""
function peturb(θ::AbstractVector, d::Sampleable; n::Integer = 1)
    (rand(d, n) .+ θ)'
end

function peturb(
    θ::AbstractVector,
    d::Sampleable,
    valid_params::Function;
    n::Integer = 1,
    )
    θᵢ = θ
    θ = Matrix{Float64}(undef, n, length(d))
    for i in 1:n
        valid = false
        attempts = 0
        while !valid
            θ′ = vec(peturb(θᵢ, d))
            valid = valid_params(θ′)
            if valid
                θ[i, :] .= θ′
            else
                attempts += 1
                if attempts == 1000
                    error("Could not find valid peturbed θ.")
                end
            end
        end
    end
    θ
end


"""
Stacks a vector of consitently sized arrays to make a new array with
dimensions (length(x), dim(x[1])...).

$(SIGNATURES)
"""
function stack_arrays(x::Vector)
    @assert all(size(x[1]) == size(el) for el in x)
    dims = (length(x), size(x[1])...)
    type = typeof(x[1][1])

    d = Array{type}(undef, dims)
    colons = fill(:, ndims(x[1]))
    for i in 1:length(x)
        d[i, colons...] = x[i]
    end
    return d
end


# Get the name of a variable
macro varname(var)
   return string(var)
end


"""
remove columns that have zero variance
$(SIGNATURES)
"""
function remove_invariant(s, s_true)
    no_var = var.(eachcol(s)) .≈ 0
    if any(no_var)
        if sum(no_var) == size(s, 2)
            error("None of the summary statistics had any variance.")
        end
        @debug """
        $(@varname(s)) has zero variance columns at index
        $(findall(no_var)). Removing these columns.
        """
        s = s[:, .!no_var]
        s_true = s_true[.!no_var]
    end
    return s, s_true
end


"""
Standardize to zero mean and standard deviation 1. Can also provide a vector
which will be standardized using the mean and standard deviation of the matrix.
$(SIGNATURES)
"""
function standardize(X::AbstractMatrix)
  means = mean.(eachcol(X))
  sds = std.(eachcol(X))
  X = (X .- means') ./ sds'
  X, means, sds
end

"""
Standardize matrix and vector, using the mean and standard deviation of the matrix.
$(SIGNATURES)
"""
function standardize(X::AbstractMatrix, y::AbstractVector)
  means = mean.(eachcol(X))
  sds = std.(eachcol(X))
  X = (X .- means') ./ sds'
  y = (y - means) ./ sds
  X, y
end


## For testing:

"""
Deterministic simulator useful for testing.
"""
function deterministic_test_simulator(θ::AbstractVector{Float64})
    @assert length(θ) == 2
    [θ[1], θ[1]*θ[2], θ[2]^2]
end

"""
Gets the analytic posterior distribution from a normal prior and normal
likelihood. Useful for testing.
"""
function analytic_mvn_posterior(
  prior::AbstractMvNormal,
  likelihood::AbstractMvNormal
  )
  Σ1 = cov(prior); μ1 = mean(prior)
  Σ2 = cov(likelihood); μ2 = mean(likelihood)
  Σ = (Σ1^-1 + Σ2^-1)^-1
  μ = Σ*Σ1^-1*μ1 + Σ*Σ2^-1*μ2
  MvNormal(μ, Σ)
end

"""
Convert covariance matrix to correlation matrix. Returns tuple (R, σ²)
"""
function cov_to_cor(Σ::Union{Diagonal, Symmetric})
    σ² = diag(Σ)
    σ = .√σ²
    R = Symmetric(diagm(1 ./ σ) * Σ * diagm(1 ./ σ))
    R, σ²
end

"""
Convert correlation matrix to covariance matrix.
"""
function cor_to_cov(R::AbstractMatrix, σ²::AbstractVector)
    σ = .√σ²
    Symmetric(diagm(σ) * R * diagm(σ))
end


"""
Used to store and print information about an object. Useful for printing neat
debugging messages.
"""
Base.@kwdef mutable struct ObjectSummaryLogger
  summaries::Vector{Function}
  data::Array{Any} = Matrix{Any}(undef, 0, length(summaries)+1)
end

function add_log!(logger::ObjectSummaryLogger, tag::String, M::AbstractMatrix)
  row = [f(M) for f in logger.summaries]
  row = permutedims([tag; row])
  logger.data = [logger.data; row]
end

function get_pretty_table(logger::ObjectSummaryLogger)
  colnames = [string(f) for f in logger.summaries]
  colnames = ["tag"; colnames]
  pretty_table(logger.data, colnames)
end


"""
Find outlier rows in matrix, using threshold multiples of iqr above and
below the median for each column.
"""
function outlier_rows(A::AbstractMatrix; iqr_tol::Float64 = 4.)
    iqrs = iqr.(eachcol(A))
    medians = median.(eachcol(A))
    lower_lim = medians .- iqr_tol*iqrs
    upper_lim = medians .+ iqr_tol*iqrs
    outliers = (A .< lower_lim') .| (A .> upper_lim')
    outlier_rows = any.(eachrow(outliers))
    outlier_rows
end


"""
Remove outlier rows from both matrices, using first matrix to determine
outliers.
"""
function rm_outliers(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    outliers = outlier_rows(A; kwargs...)
    A[.!outliers, :], B[.!outliers, :]
end

"""
Remove outlier rows from matrix.
"""
function rm_outliers(A::AbstractMatrix; kwargs...)
    outliers = outlier_rows(A; kwargs...)
    A[.!outliers, :]
end
