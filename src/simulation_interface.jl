# Contains general functoins for the simulation interface

"""
Simulates summary statistics from the model under a fixed parameter vector,
or a matrix of parameter vectors. If a vector is used, `n_sim` is specified as
the number of simulations. If a matrix is used, one simulation is carried out
using each row as a parameter vector. By defualt simulations are run on multiple
threads (see Threads manual for information on starting Julia with multiple
threads).

$(SIGNATURES)

# Arguments
- `θ::AbstractVector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator.
- `n_sim::Int` Number of simulations.
- `simulator_kwargs` Kwargs passed to simulator.
- `summary_kwargs` Kwargs passed to summary.
- `parallel::Bool = true` Wheter to run on multiple threads.
"""
function simulate_n_s(
    θ::AbstractVector;
    simulator::Function,
    summary::Function,
    n_sim::Int,
    simulator_kwargs = Dict(),
    summary_kwargs = Dict(),
    parallel::Bool = true
    )

    # First simulation outside for loop to get s length
    x = simulator(θ; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results = Array{Float64}(undef, n_sim, length(s))
    results[1, :] = s

    function fill_results!(i)
        x = simulator(θ, simulator_kwargs...)
        s = summary(x, summary_kwargs...)
        results[i, :] = s
    end

    if parallel
        Threads.@threads for i in 2:n_sim
            fill_results!(i)
        end
    else
        for i in 2:n_sim
            fill_results!(i)
        end
    end

    return results
end


"""
As for above, but uses an Array of parameter values are used, carrying out one
    simulation from each row of θ.

    $(SIGNATURES)
"""
function simulate_n_s(
    θ::AbstractMatrix;
    simulator::Function,
    summary::Function,
    simulator_kwargs = Dict(),
    summary_kwargs = Dict(),
    parallel::Bool = true
    )

    # First simulation outside for loop to get s length
    x = simulator(θ[1, :]; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results = Array{Float64}(undef, size(θ)[1], length(s))
    results[1, :] = s

    function fill_results!(i)
        x = simulator(θ[i, :], simulator_kwargs...)
        s = summary(x, summary_kwargs...)
        results[i, :] = s
    end

    if parallel
        Threads.@threads for i in 2:size(θ)[1]
            fill_results!(i)
        end

    else
        for i in 2:size(θ)[1]
            fill_results(i)
        end
    end

    return results
end
