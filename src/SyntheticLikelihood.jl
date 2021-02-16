module SyntheticLikelihood

function test_func()
    return true
end

using Distributions



# TODO Add basic simulator? but probably want seperate script?


"""
Evaluates a point estimate of the synthetic likelihood.
"""
function synthetic_likelihood(;
    θ::Vector,
    simulator::Function,
    summary::Function,
    s_true::Vector,
    n_sim::Int,
    simulator_kwargs,
    summary_kwargs,
    )

    x = simulator(θ, simulator_kwargs...)
    s = summary(x, summary_kwargs...)

    sum_stats = sim_sum(θ, simulator, sum_stats, n_s, n_sims)
    μ = mean.(eachcol(sum_stats))
    Σ = cov(sum_stats)
    mvn = MultivariateNormal(μ, Σ)

    return logpdf(mvn, s_true)
end

length([1,2,3])

"""
Simulates summary statistics from the model under a fixed parameter vector.

# Arguments
- `θ::Vector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator.
- `n_sim::Int` Number of simulations.
- `simulator_kwargs` Kwargs passed to simulator.
- `summary_kwargs` Kwargs passed to summary.

"""
function simulate_n_s(;
    θ::Vector,
    simulator::Function,
    summary::Function,
    n_sim::Int,
    simulator_kwargs,
    summary_kwargs,
    )

    # First simulation outside for loop to get s length
    x = simulator(θ, simulator_kwargs...)
    s = summary(x, summary_kwargs...)

    results::Array{Float64}(undef, n_sim, length(s))
    results[1, :] = s

    Threads.@threads for i in 2:n_sim
        x = simulator(θ, simulator_kwargs...)
        s = summary(x, summary_kwargs...)
        results[i, :] = s
    return results
end

"""
Simulates summary statistics from a matrix of parameter vectors.


"""
function simulate_n_s(;
    θ::Array,
    simulator::Function,
    summary::Function,
    )
    error("unimplemented")
    # TODO: not implemented
end

# TODO delete below
function sim_sum(θ::Vector, simulator::Function,
    sum_stats::Function, n_s::Int, n_sims::Int=1)

    m = Array{Float64}(undef, n_sims, n_s)
    for i in 1:n_sims
        sum_vec = simulator(θ) |> sum_stats
        m[i, :] = sum_vec
    end
    return m
end
