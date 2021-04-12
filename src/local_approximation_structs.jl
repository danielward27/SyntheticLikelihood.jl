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
    init_P::AbstractMvNormal
    "The number of peturbed points to use for the local regression."
    n_sim::Integer
    "Adaptive proposal distribution. Should not be set manually."
    P::AbstractMvNormal = init_P
    P_regularizer = KitchenSink(ref = cov(init_P))
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
    init_P::AbstractMvNormal = MvNormal(cov(prior))
    n_sim::Integer
    P::AbstractMvNormal = init_P
    P_regularizer = KitchenSink(ref = cov(prior))
    "Prior distribution (either multivariate or Product distribution)"
end
