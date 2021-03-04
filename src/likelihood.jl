
"""
Evaluates synthetic likelhood of observed data for a fixed parameter vector
    using a multivariate Gaussian assumption as in (Simon Wood, 2010).

    $(SIGNATURES)

# Arguments
- `θ::AbstractVector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator (defualt identity).
- `s_true::AbstractVector` Observed summary statistics.
- `n_sim::Integer` Number of simulations to use.
- `simulator_kwargs` Kwargs splatted in simulator.
- `summary_kwargs` Kwargs splatted in summary.
"""
function synthetic_likelihood(
    θ::AbstractVector;
    simulator::Function,
    summary::Function = identity,
    s_true::AbstractVector,
    n_sim::Integer,
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
