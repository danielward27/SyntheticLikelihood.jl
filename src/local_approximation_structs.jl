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
    "Method to regularise the inverse Hessian to a reasonable covariance matrix."
    P_regularizer::AbstractRegularizer = KitchenSink(ref = Symmetric(Matrix(cov(P))))
    """
    Parameter constraints. Function that takes a parameter vector and returns
    true (for valid parameters) or false (for invalid parameters). Defualt
    is θ -> true
    """
    valid_params::Function = θ -> true
end


"""
Contains the hyperparameters for getting a local approximation
of the posterior. In contrast to the likelihood version, a prior is provided,
P is by defualt a MvNormal with covariance 0.5*cov(prior), and 
valid_params checks whether proposed points fall within the prior support.

$(FIELDS)
"""
Base.@kwdef mutable struct LocalPosterior <: LocalApproximation
    "Prior distribution (either multivariate or Product distribution)"
    prior::Sampleable
    simulator::Function
    summary::Function=identity
    s_true::Vector{Float64}
    P::AbstractMvNormal = MvNormal(0.5.*cov(prior))
    n_sim::Integer = 1000
    P_regularizer::AbstractRegularizer = KitchenSink(ref = Symmetric(Matrix(0.5.*cov(prior))))
    valid_params::Function = θ -> insupport(prior, θ)
end
