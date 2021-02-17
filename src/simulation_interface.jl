# Contains general functoins for the simulation interface

"""
Simulates summary statistics from the model under a fixed parameter vector,
or an array of parameter vectors. If a vector is used, `n_sim` is specified as
the number of simulations. If an array is used, one simulation is carried out
using each row as a parameter vector.

$(SIGNATURES)

# Arguments
- `θ::Vector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator.
- `n_sim::Int` Number of simulations.
- `simulator_kwargs` Kwargs passed to simulator.
- `summary_kwargs` Kwargs passed to summary.
"""
function simulate_n_s(
    θ::Vector{Float64};
    simulator::Function,
    summary::Function,
    n_sim::Int,
    simulator_kwargs = Dict(),
    summary_kwargs = Dict(),
    )

    # First simulation outside for loop to get s length
    x = simulator(θ; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results = Array{Float64}(undef, n_sim, length(s))
    results[1, :] = s

    Threads.@threads for i in 2:n_sim
        x = simulator(θ, simulator_kwargs...)
        s = summary(x, summary_kwargs...)
        results[i, :] = s
    end
    return results
end


"""
As for above, but uses a matrix of parameter values, carrying out one simulation
    from each row of θ.

    $(SIGNATURES)
"""
function simulate_n_s(
    θ::Array{Float64, 2};
    simulator::Function,
    summary::Function,
    simulator_kwargs = Dict(),
    summary_kwargs = Dict()
    )

    # First simulation outside for loop to get s length
    x = simulator(θ[1, :]; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results = Array{Float64}(undef, size(θ)[1], length(s))
    results[1, :] = s

    Threads.@threads for i in 2:size(θ)[1]
        x = simulator(θ[i, :], simulator_kwargs...)
        s = summary(x, summary_kwargs...)
        results[i, :] = s
    end
    return results
end
