# utility functions

using Distributions

"""
Simulates from MVN normal to act as a simple test example.
"""
function test_simulator(θ)
    d = MvNormal(θ)
    rand(d)
end

"""
Passes the x values straight through.
"""
function test_summary(x)
    return x
end
