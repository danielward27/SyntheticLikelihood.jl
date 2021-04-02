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

# Same as above but has prior
"""
Contains the hyperparameters for getting a local approximation
of the posterior (using `LocalLikelihood` and a prior).

$(FIELDS)
"""
Base.@kwdef mutable struct LocalPosterior <: LocalApproximation
    "Prior distribution."
    prior::Sampleable
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

## Make it simple to convert between the two.
function LocalPosterior(prior::Sampleable, local_likelihood::LocalLikelihood)
    ll = local_likelihood
    LocalPosterior(
        prior, ll.simulator, ll.summary, ll.s_true,
        ll.P, ll.n_sim, ll.eigval_threshold
    )
end

function LocalLikelihood(local_posterior::LocalPosterior)
    lp = local_posterior
    LocalLikelihood(
        lp.simulator, lp.summary, lp.s_true,
        lp.P, lp.n_sim, lp.eigval_threshold
        )
end
