# utility functions

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
Pairwise combinations (for quadratic regression). `n=5` would return all the
pairwise combinations between 1:5 (including matched terms e.g. [1,1]).

$(SIGNATURES)

"""
function pairwise_combinations(n::Integer)
    n_combinations = binomial(n, 2) + n
    combinations = Matrix{Int64}(undef, n_combinations, 2)
    row = 1
    for i in 1:n
        j = 1
        while j <= i
            combinations[row, :] = [j, i]
            row += 1
            j +=1
        end
    end
    combinations
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


## For testing:
# Deterministic simulator for testing
function deterministic_test_simulator(θ::AbstractVector{Float64})
    @assert length(θ) == 2
    [θ[1], θ[1]*θ[2], θ[2]^2]
end
