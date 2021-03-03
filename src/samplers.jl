# Samplers to explore the likelihood

"""
Struct for containing the state of sampler at a particular iteration.
    Gradient and hessian are `missing` unless specified.
"""
Base.@kwdef mutable struct SamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::Union{AbstractVector, Missing} = missing
    hessian::Union{AbstractMatrix, Missing} = missing
    counter::Integer = 0
end


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



abstract type AbstractSampler end

## Langevin diffusion
"""
Sampler object for Langevin diffusion. Uses a discrete time Euler approximation of
the Langevin diffusion (unadjusted Langevin algorithm), given by the update
θ := θ - η/2 * ∇ + ξ, where ξ is given by N(0, ηI).

$(SIGNATURES)
"""
Base.@kwdef mutable struct Langevin <: AbstractSampler
    step_size::Float64
    objective::Function
    gradient::Function
end


function get_init_state(
    sampler::Langevin,
    init_θ::Vector{Float64})

    SamplerState(;θ = init_θ,
                objective = sampler.objective(init_θ),
                gradient = sampler.gradient(init_θ))
end

function update!(sampler::Langevin, state::SamplerState)
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient
    ξ = rand(MvNormal(length(θ) , sqrt(η)))
    state.θ = θ .- η ./ 2 .* ∇ .+ ξ
    state.objective = sampler.objective(θ)
    state.gradient = sampler.gradient(θ)
    state.counter += 1
end





## Preconditioned Langevin diffusion
"""
Sampler object for Preconditioned Langevin diffusion. Also can be thought of as
    a stochastic newton method. Uses the update:
    θ := θ - η/2 * H⁻¹*∇ + ξ, where ξ ∼ N(0, ηH⁻¹).

    $(SIGNATURES)
"""
Base.@kwdef mutable struct PreconditionedLangevin <: AbstractSampler
    step_size::Float64
    objective::Function
    gradient_hessian::Function
end

function get_init_state(
    sampler::PreconditionedLangevin,
    init_θ::Vector{Float64})

    gradient, hessian = sampler.gradient_hessian(init_θ)
    SamplerState(;θ = init_θ,
                objective = sampler.objective(init_θ),
                gradient, hessian)
end

function update!(sampler::PreconditionedLangevin, state::SamplerState)
    η, θ, ∇, H  = sampler.step_size, state.θ, state.gradient, state.hessian
    ξ = rand(MvNormalCanon(1/η .* H))  # Equiv to N(0, η.*H⁻¹)
    H = Symmetric(H)
    state.θ = θ .- η/2 .* (H \ ∇) .+ ξ
    state.objective = sampler.objective(θ)
    state.gradient = sampler.gradient(θ)
    state.counter += 1
end



"""
Run the sampling algorithm. Data to collect at each iteration is specified by
`collect_data`, and should be a subset of
`[:θ, :objective, :gradient, :hessian, :counter]`.

$(SIGNATURES)

Returns a tuple, with keys matching `collect_data`.
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
