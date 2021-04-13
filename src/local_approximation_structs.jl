abstract type LocalApproximation end

"""
Contains the hyperparameters for getting a local approximation
of the negative log-likelihood surface using local regressions.

$(FIELDS)
"""
Base.@kwdef mutable struct LocalLikelihood <: LocalApproximation
    simulator::Function
    summary::Function=identity
    s_true::Vector{Float64}
    "Initial distribution used to peturb the parameter value."
    P::AbstractMvNormal
    "The number of peturbed points to use for the local regression."
    n_sim::Integer = 1000
    "Adaptive proposal distribution. Should not be set manually."
    P_regularizer::AbstractRegularizer = KitchenSink(ref = Symmetric(cov(P)))
end


"""
Contains the hyperparameters for getting a local approximation
of the posterior.

$(FIELDS)
"""
Base.@kwdef mutable struct LocalPosterior <: LocalApproximation
    prior::Sampleable
    simulator::Function
    summary::Function=identity
    s_true::Vector{Float64}
    P::AbstractMvNormal = MvNormal(cov(prior))
    n_sim::Integer = 1000
    P_regularizer::AbstractRegularizer = KitchenSink(ref = Symmetric(cov(prior)))
    "Prior distribution (either multivariate or Product distribution)"
end
