module SyntheticLikelihood

using Distributions
using DocStringExtensions
using Parameters
using LinearAlgebra

include("simulation_interface.jl")
include("likelihood.jl")
include("mcmc.jl")
include("utils.jl")
include("local_regression.jl")


# simulation interface
export simulate_n_s

# likelihood
export synthetic_likelihood

# mcmc

# utils
export peturb

end
