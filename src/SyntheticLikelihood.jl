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
using StatsBase
using DelimitedFiles
using ProgressMeter
import Distributions: cov, insupport, logpdf

include("matrix_regularizers.jl")
include("prior.jl")
include("local_approximation_structs.jl")
include("quadratic_local_regression.jl")
include("glm_local_regression.jl")
include("local_regression.jl")
include("samplers.jl")
include("utils.jl")
include("data_collector.jl")
include("simulate_n_s.jl")

# Matrix regularization
export KitchenSink, regularize

# Prior
export Prior, sample_θ

# likelihood
export synthetic_likelihood

# Local approximation structs
export LocalApproximation, LocalLikelihood, LocalPosterior, BasicPosterior

# Sampling
export AbstractSampler, ULA, RiemannianULA, RWMetropolis, run_sampler!

# plotting
export plot_prior_posterior_density

end
