# Evaluate likelihood (and its gradient/hessian)

"""
Evaluates synthetic likelhood of observed data for a fixed parameter vector
    using a multivariate Gaussian assumption as in (Simon Wood, 2010).

# Arguments
- `θ::Vector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator.
- `s_true::Vector` Observed summary statistics.
- `n_sim::Int` Number of simulations to use.
- `simulator_kwargs` Kwargs passed to simulator.
- `summary_kwargs` Kwargs passed to summary.
"""
function synthetic_likelihood(
    θ::Vector;
    simulator::Function,
    summary::Function,
    s_true::Vector,
    n_sim::Int,
    simulator_kwargs,
    summary_kwargs,
    )

    x = simulator(θ; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    sum_stats = sim_sum(θ, simulator, sum_stats, n_s, n_sims)
    μ = mean.(eachcol(sum_stats))
    Σ = cov(sum_stats)
    mvn = MultivariateNormal(μ, Σ)

    return logpdf(mvn, s_true)
end
