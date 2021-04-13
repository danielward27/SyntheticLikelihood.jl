module SyntheticLikelihood

using Distributions
using DocStringExtensions
using LinearAlgebra
using Statistics
using GLM
using ForwardDiff
using Parameters
using PrettyTables
import Base.@kwdef

include("matrix_regularizers.jl")
include("local_approximation_structs.jl")
include("quadratic_local_regression.jl")
include("glm_local_regression.jl")
include("local_regression.jl")
include("samplers.jl")
include("utils.jl")
include("data_collector.jl")
include("simulate_n_s.jl")

# simulate n s
export simulate_n_s

# Matrix regularization
export KitchenSink, Flip, regularize

# likelihood
export synthetic_likelihood

# local regression
export Localμ, quadratic_local_μ, LocalΣ, glm_local_Σ, local_likelihood,
    ObjGradHess, obj_grad_hess, LocalApproximation, LocalLikelihood, LocalPosterior

# Samplers
export ULA, RiemannianULA, run_sampler!

# utils
export peturb

end
