
"""
Peturb a vector using a user specified distribution (often MVN zero mean).
Returns array of size (n, length(θ))

$(SIGNATURES)

# Arguments
- `θ::AbstractVector` Parameter to peturb.
- `d::Sampleable` Distribution from which to sample (see Distributions.jl).
-  `n::Integer = 1` Number of peturbed vectors to return.
"""
function peturb(θ::AbstractVector, d::Sampleable, n::Integer = 1)
    (rand(d, n) .+ θ)'
end


"""
Stacks a vector of consitently sized arrays to make a new array with
dimensions (length(x), dim(x[1])...).
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


# Used to remove statistics that have zero variance
function remove_invariant(s; warn=true)
    no_var = var.(eachcol(s)) .≈ 0
    if any(no_var)
        if warn
            @warn "$(@varname(s)) has zero variance columns at index "*
            "$(findall(no_var)). Removing these columns."
        end
        s = s[:, .!no_var]
    end
    return s
end


"""
Make matrix positive definite using eigen decomposition.
Flips negative eiugnvalues then ensure all greater than threshold.
"""
function ensure_posdef(M::Symmetric, threshold::Float64)
  M = eigen(M)
  vals = abs.(M.values)
  vals[vals .< threshold] .= threshold
  M = Eigen(vals, M.vectors)
  Symmetric(Matrix(M))
end



## For testing:
# Deterministic simulator for testing
function deterministic_test_simulator(θ::AbstractVector{Float64})
    @assert length(θ) == 2
    [θ[1], θ[1]*θ[2], θ[2]^2]
end
