module SyntheticLikelihood

using Distributions
using DocStringExtensions
using LinearAlgebra
using Statistics
using GLM


include("simulation_interface.jl")
include("likelihood.jl")
include("mcmc.jl")
include("utils.jl")
include("local_regression.jl")


# simulation interface
export simulate_n_s

# likelihood
export synthetic_likelihood

# local regression
export quadratic_design_matrix
export linear_regression
export Localμ
export quadratic_local_μ
export get_residuals
export LocalΣ
export glm_local_Σ

# mcmc

# utils
export peturb
export pairwise_combinations

end
