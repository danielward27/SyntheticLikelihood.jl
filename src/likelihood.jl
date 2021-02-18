# Evaluate likelihood (and its gradient/hessian)

"""
Evaluates synthetic likelhood of observed data for a fixed parameter vector
    using a multivariate Gaussian assumption as in (Simon Wood, 2010).

    $(SIGNATURES)

# Arguments
- `θ::Vector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator.
- `s_true::Vector` Observed summary statistics.
- `n_sim::Int` Number of simulations to use.
- `simulator_kwargs` Kwargs splatted in simulator.
- `summary_kwargs` Kwargs splatted in summary.
"""
function synthetic_likelihood(
    θ::Vector;
    simulator::Function,
    summary::Function,
    s_true::Vector,
    n_sim::Int,
    simulator_kwargs = NamedTuple(),
    summary_kwargs = NamedTuple(),
    )

    s = simulate_n_s(θ; simulator, summary, n_sim,
                             simulator_kwargs, summary_kwargs)

    μ = mean.(eachcol(s))
    Σ = cov(s)
    mvn = MultivariateNormal(μ, Σ)

    return logpdf(mvn, s_true)
end
