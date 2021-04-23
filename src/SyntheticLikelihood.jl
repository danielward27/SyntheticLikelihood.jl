module SyntheticLikelihood

using StatsPlots
using Distributions
using DocStringExtensions
using LinearAlgebra
using Statistics
using GLM
using ForwardDiff
using Parameters
using PrettyTables
import Base.@kwdef
using StatsBase
using DelimitedFiles
using ProgressMeter

include("matrix_regularizers.jl")
include("local_approximation_structs.jl")
include("quadratic_local_regression.jl")
include("glm_local_regression.jl")
include("local_regression.jl")
include("samplers.jl")
include("utils.jl")
include("data_collector.jl")
include("simulate_n_s.jl")
include("plotting.jl")


# Matrix regularization
export KitchenSink, regularize

# likelihood
export synthetic_likelihood

# Local approximation structs
export LocalApproximation, LocalLikelihood, LocalPosterior

# Sampling
export AbstractSampler, ULA, RiemannianULA, run_sampler!

# plotting
export plot_prior_posterior_density

end
