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

peturb([1,2,3], [1,2,3], 2)








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
