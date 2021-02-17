module SyntheticLikelihood

using Distributions

function test_func()
    return true
end


# TODO Add basic simulator? but probably want seperate script?



"""
Evaluates a point estimate of the synthetic likelihood.

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
function simulate_n_s(
    θ::Vector;
    simulator::Function,
    summary::Function,
    n_sim::Int,
    simulator_kwargs,
    summary_kwargs,
    )

    # First simulation ouget tside for loop to get s length
    x = simulator(θ; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results::Array{Float64}(undef, n_sim, length(s))
    results[1, :] = s

    Threads.@threads for i in 2:n_sim
        x = simulator(θ, simulator_kwargs...)
        s = summary(x, summary_kwargs...)
        results[i, :] = s
    end
    return results
end

"""
Simulates summary statistics from a matrix of parameter vectors.

# Arguments
- `θ::Array` Array of parameters. Each row is passed to simulator.
- `simulator::Function` Simulator function.
- `summary::Function` Summary function that takes output of simulator.
- `simulator_kwargs` Kwargs passed to simulator.
- `summary_kwargs` Kwargs passed to summary.

"""
function simulate_n_s(
    θ::Array{Any, 2};
    simulator::Function,
    summary::Function,
    simulator_kwargs,
    summary_kwargs
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

end
