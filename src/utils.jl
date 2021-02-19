# utility functions


"""
Peturb parameter vector with multivariate normal. Returns matrix
of size [n, length(θ)].
"""
function peturb(θ::AbstractVector, Σ::AbstractMatrix, n::Int = 1)
    @assert length(θ) == size(Σ)[2]
    d = MvNormal(θ, Σ)
    rand(d, n)'
end


"""
Same as above but uses diagonal covariance specified with vector
"""
function peturb(θ::AbstractVector, Σ::AbstractVector, n::Int = 1)
    @assert length(θ) == length(Σ)
    d = MvNormal(θ, sqrt.(Σ))
    rand(d, n)'
end


# Pairwise combinations (for polynomial regression)
function pairwise_combinations(n::Int)
    n_combinations = binomial(n, 2) + n
    combinations = Matrix{Int64}(undef, n_combinations, 2)
    row = 1
    for i in 1:n
        for j in 1:n
            if i <= j
                combinations[row, :] = [i, j]
                row += 1
            end
        end
    end
    combinations
end



## For testing:

# Simulates from MVN normal to act as a simple test example.
function test_simulator(θ::AbstractVector; var = nothing)
    if isnothing(var)
        var = ones(length(θ))
    end
    d = MvNormal(θ, var)  # Diagonal covariance
    rand(d)
end

# Passes the x values straight through.
function test_summary(x::AbstractVector)
    return x
end
