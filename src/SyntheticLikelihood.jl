module SyntheticLikelihood

using Distributions
using DocStringExtensions
using LinearAlgebra
using Statistics
using GLM

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
export quadratic_design_matrix, linear_regression, Localμ, quadratic_local_μ,
 get_residuals, LocalΣ, glm_local_Σ, local_synthetic_likelihood, ObjGradHess

# Samplers
export Langevin, PreconditionedLangevin, run_sampler!

# utils
export peturb

end
