abstract type LocalApproximation end

"""
Contains the hyperparameters for getting a local approximation
of the negative log-likelihood surface using local regressions.

$(FIELDS)
"""
Base.@kwdef mutable struct LocalLikelihood <: LocalApproximation
    "The simulator function taking parameter vector θ."
    simulator::Function
    "The summary function taking output from the simulator."
    summary::Function=identity
    "The observed summary statistics."
    s_true::Vector{Float64}
    "Distribution used to peturb the parameter value."
    P::AbstractMvNormal
    "The number of peturbed points to use for the local regression."
    n_sim::Integer
    """Minimum eigenvalue threshold for the estimated hessian. Negative
    eigenvalues are flipped and those smaller than the threshold are
    set to the threshold."""
    eigval_threshold::Float64 = 0.2
end


"""
Contains the hyperparameters for getting a local approximation
of the posterior (using `LocalLikelihood` and a prior).

$(FIELDS)
"""
Base.@kwdef mutable struct LocalPosterior <: LocalApproximation
    prior::Sampleable
    local_likelihood::LocalLikelihood
end

# Below needed to allow dispatch on abstract type LocalApproximation.
LocalLikelihood(local_likelihood::LocalLikelihood) = local_likelihood
LocalLikelihood(local_posterior::LocalPosterior) = local_posterior.local_likelihood
