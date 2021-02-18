# utility functions

using Distributions

# Simulates from MVN normal to act as a simple test example.
function test_simulator(θ::Vector{Float64}; var = nothing)
    if isnothing(var)
        var = ones(length(θ))
    end
    d = MvNormal(θ, var)  # Diagonal covariance
    rand(d)
end

# Passes the x values straight through.
function test_summary(x::Vector{Float64})
    return x
end
