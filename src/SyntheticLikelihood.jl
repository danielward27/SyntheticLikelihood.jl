module SyntheticLikelihood

using Distributions
using DocStringExtensions
using LinearAlgebra
using Statistics
using GLM
using ForwardDiff

import Base.@kwdef

include("local_approximation_structs.jl")
include("quadratic_local_regression.jl")
include("glm_local_regression.jl")
include("local_regression.jl")
include("samplers.jl")
include("utils.jl")
include("data_collector.jl")
include("simulate_n_s.jl")

# simulation interface
export simulate_n_s

# likelihood
export synthetic_likelihood

# local regression
export Localμ, quadratic_local_μ, LocalΣ, glm_local_Σ, local_likelihood,
    ObjGradHess, LocalApproximation, LocalLikelihood, LocalPosterior

# Samplers
export Langevin, PreconditionedLangevin, run_sampler!

# utils
export peturb

end
