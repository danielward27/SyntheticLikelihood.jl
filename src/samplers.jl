# Samplers to explore the likelihood


## Helper functions to collect data
"""
Function initialises a named tuple containing Vectors with undefined values.
Used with samplers to store results. State just provides an "example" state
from which to infer types of vectors in the array. Names of the named tuple
are the symbols provided.
"""
function init_data_tuple(
    state::SamplerState,
    collect_data::Vector{Symbol},
    n_steps::Integer)

    names = collect_data
    values = Vector{Array}(undef, length(names))

    for (i, symbol) in enumerate(collect_data)
        values[i] = Vector{typeof(getproperty(state, symbol))}(undef, n_steps)
    end
    (;zip(names, values)...)
end


"""
Add data to the data tuple.
"""
function add_state!(
    data::NamedTuple, state::SamplerState, idx::Integer)
    for symbol in keys(data)
        field = getproperty(state, symbol)
        data[symbol][idx] = field
    end
    data
end


"""
Loop through named tuple and call stack_arrays on any vector whose
elements are an array. Used at end of samplers.
"""
function simplify_data(data::NamedTuple)
    symbols = keys(data)
    new_values = Vector{Array}(undef, length(symbols))

    for (i, x) in enumerate(data)
        if x[1] isa Array
            new_values[i] = stack_arrays(x)
        else
            new_values[i] = x
        end
    end
    (;zip(symbols, new_values)...)
end



## Sampling algorithms


"""
Struct for containing the state of sampler at a particular iteration.
"""
Base.@kwdef mutable struct SamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::AbstractVector
    hessian = I
    counter::Integer = 0
end

abstract type AbstractSampler end

"""
Sample using Langevin diffusion . Uses a discrete time Euler approximation of
the Langevin diffusion (unadjusted Langevin algorithm), given by the update
update θ := θ - η/2 .* ∇θ .+ ξ. ξ is given by `MvNormal(η)`. Uses a fixed step size.
"""
Base.@kwdef mutable struct Langevin <: AbstractSampler
    step_size::Vector{Float64}
    objective::Function
    gradient::Function
end


function update!(sampler::Langevin, state::SamplerState)
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient
    ξ = rand(MvNormal(η))
    state.θ = θ .- η ./ 2 .* ∇ .+ ξ
    state.objective = sampler.objective(θ)
    state.gradient = sampler.gradient(θ)
    state.counter += 1
end

function get_init_state(
    sampler::Langevin,
    init_θ::Vector{Float64})

    SamplerState(;θ = init_θ,
                objective = sampler.objective(init_θ),
                gradient = sampler.gradient(init_θ))
end


"""
Run the sampling algorithm.
"""
function run_sampler!(
    sampler::AbstractSampler,
    init_θ::Vector{Float64},
    n_steps::Integer,
    collect_data::Vector{Symbol} = [:θ, :objective])

    state = get_init_state(sampler, init_θ)
    data = init_data_tuple(state, collect_data, n_steps)

    for i in 1:n_steps
        update!(sampler, state)
        add_state!(data, state, i)
    end
    simplify_data(data)
end


mutable struct PreconditionedLangevin
    step_size::Vector{Float64}
    gradient_hessian::Function  # must return tuple of gradient and hessian
end

function update!(sampler::PreconditionedLangevin, state::SamplerState)
    error("unimplemented error")
    MvCanon
end
