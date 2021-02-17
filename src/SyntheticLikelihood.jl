module SyntheticLikelihood

using Distributions

include("utils.jl")
include("simulation_interface.jl")
include("algorithms.jl")

# Exports
export synthetic_likelihood
export simulate_n_s

end
