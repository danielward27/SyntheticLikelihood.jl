module SyntheticLikelihood

using Distributions
using DocStringExtensions
using LinearAlgebra
using Statistics
using GLM
using ForwardDiff

import Base.@kwdef

include("simulation_interface.jl")
include("likelihood.jl")
include("samplers.jl")
include("utils.jl")
include("local_regression.jl")


# simulation interface
export simulate_n_s

# likelihood
export synthetic_likelihood

# local regression
export Localμ, quadratic_local_μ, LocalΣ, glm_local_Σ, local_likelihood,
    ObjGradHess, LocalLikelihood, LocalPosterior

# Samplers
export Langevin, PreconditionedLangevin, run_sampler!

# utils
export peturb

end
