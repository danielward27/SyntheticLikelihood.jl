# Contains general functoins for the simulation interface
"""
Simulates summary statistics from the model under a fixed parameter vector.
`n_sim` is specified as the number of simulations. Simulations can be run on
multiple threads using `parallel = false`. By defualt no summary statistic function
is used (by passing the `identity` function).

$(SIGNATURES)

# Arguments
- `θ::AbstractVector` Parameter vector passed to simulator.
- `simulator::Function` Simulator.
- `summary::Function` Summary function that takes output of simulator (defualt identity).
- `n_sim::Integer` Number of simulations.
- `simulator_kwargs` Kwargs passed to simulator.
- `summary_kwargs` Kwargs passed to summary.
- `parallel::Bool = false` Whether to run on multiple threads.
"""
function simulate_n_s(
    θ::AbstractVector;
    simulator::Function,
    summary::Function = identity,
    n_sim::Integer = 1,
    simulator_kwargs = Dict(),
    summary_kwargs = Dict(),
    parallel::Bool = false
    )

    # First simulation outside for loop to get s length
    x = simulator(θ; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results = Array{Float64}(undef, n_sim, length(s))
    results[1, :] = s

    if parallel
        # Note storing intermediate results naively could create a data race
        Threads.@threads for i in 2:n_sim
            results[i, :] = summary(simulator(θ; simulator_kwargs...),
                                    summary_kwargs...)
        end
    else
        for i in 2:n_sim
            x = simulator(θ; simulator_kwargs...)
            s = summary(x; summary_kwargs...)
            results[i, :] = s
        end
    end

    return results
end


"""
As for above, but a Matrix of parameter values are used, carrying out one
    simulation from each row of θ (and hence `n_sim` is not required).

    $(SIGNATURES)
"""
function simulate_n_s(
    θ::AbstractMatrix;
    simulator::Function,
    summary::Function = identity,
    simulator_kwargs = Dict(),
    summary_kwargs = Dict(),
    parallel::Bool = false
    )

    # First simulation outside for loop to get s length
    x = simulator(θ[1, :]; simulator_kwargs...)
    s = summary(x; summary_kwargs...)

    results = Array{Float64}(undef, size(θ, 1), length(s))
    results[1, :] = s

    if parallel
        Threads.@threads for i in 2:size(θ, 1)
            results[i, :] = summary(simulator(θ[i, :]; simulator_kwargs...);
                                    summary_kwargs...)
                                end

    else
        for i in 2:size(θ, 1)
            x = simulator(θ[i, :]; simulator_kwargs...)
            s = summary(x; summary_kwargs...)
            results[i, :] = s
        end
    end

    return results
end
