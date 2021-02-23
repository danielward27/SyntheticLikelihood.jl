# utility functions

"""
Peturb a vector using a user specified distribution (often MVN zero mean).
Returns array of size (n, length(θ))

$(SIGNATURES)

# Arguments
- `θ::AbstractVector` Parameter to peturb.
- `d::Sampleable` Distribution from which to sample (see Distributions.jl).
-  `n::Int = 1` Number of peturbed vectors to return.
"""
function peturb(θ::AbstractVector, d::Sampleable, n::Int = 1)
    (rand(d, n) .+ θ)'
end

"""
Pairwise combinations (for quadratic regression). `n=5` would return all the
pairwise combinations between 1:5 (including matched terms e.g. [1,1]).

$(SIGNATURES)

"""
function pairwise_combinations(n::Int)
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


## For testing:

# Deterministic simulator for testing
function deterministic_test_simulator(θ::AbstractVector{Float64})
    @assert length(θ) == 2
    [θ[1], θ[1]*θ[2], θ[2]^2]
end
