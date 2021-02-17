module SyntheticLikelihood

using Distributions
using DocStringExtensions

include("utils.jl")
include("simulation_interface.jl")
include("likelihood.jl")
include("mcmc.jl")

# simulation interface
export simulate_n_s

# likelihood
export synthetic_likelihood

# mcmc

end
