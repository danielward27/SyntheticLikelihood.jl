# Samplers to explore the likelihood

## Type/struct definitions
"""
Concrete sampler types should have fields:
- θ::AbstractVector
- objective::Float64
- counter::Integer

"""
abstract type AbstractSamplerState end


"""
Struct for containing the state of sampler at each iteration
    for simple samplers (that just require objective evaluation).
"""
mutable struct BasicState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    counter::Integer
    BasicState(θ, objective) = new(θ, objective, 0)
end


"""
Struct for containing the state of sampler at each iteration
    for simple gradient based samplers.
"""
mutable struct GradientState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    counter::Integer
    gradient::AbstractVector

    function GradientState(θ, objective, gradient)
        @assert length(θ) == length(gradient)
        new(θ, objective, 0, gradient)
    end
end


"""
Struct for containing the state of sampler at each iteration
    for samplers which use the gradient and hessian.
"""
mutable struct GradientHessianState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    counter::Integer
    gradient::AbstractVector
    hessian::AbstractMatrix
    function GradientState(θ, objective, gradient, hessian)
        @assert length(θ)==length(gradient)==shape(hessian, 1)==shape(hessian, 2)
        new(θ, objective, 0, gradient)
    end
end

## Helper functions
"""
Function initialises a named tuple containing Vectors with undefined values.
Used with samplers to store results. State just provides an "example" state
from which to infer types of vectors in the array. Names of the named tuple
are the symbols provided.
"""
function init_data_tuple(
    state::AbstractSamplerState,
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
    data::NamedTuple, state::AbstractSamplerState, idx::Integer)
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

# TODO Update documentation below
# TODO Test it (and data storage)

"""
Sample using Langevin diffusion . Uses a discrete time Euler approximation of
the Langevin diffusion (unadjusted Langevin algorithm), given by the update
update θ := θ - step_size/2 .* ∇θ .+ ξ. ξ is given by `MvNormal(step_size)`.
Uses a fixed step size.

Note if the aim is to sample from a density function, the negative of its
gradient and density should be provided (sampler "aims" to minimize the function).

Returns a tuple, (data, state), where data is a NamedTuple of stored data,
and state is the state at the final iteration, which can be used to reinitialise
the algorithm if desired.

Arguments:
`state::GradientState` Initial starting state for sampler.
`objective::Function` Objective function (assumed "aim" would be to minimize)
`gradient::Function` Gradient of the objective function with respect to the parameters.
`step_size::Vector{Float64}` Multiplied elementwise by gradient.
`n_steps::Integer` Number of iterations to carry out.
`collect_data::Vector{Symbol}=[:θ, :objective]` Vector of symbols, denoting the
    items in the state to store at each iteration.

$(SIGNATURES)
"""
function langevin_diffusion(
    state::GradientState;
    objective::Function,
    gradient::Function,
    step_size::Vector{Float64},
    n_steps::Integer,
    collect_data::Vector{Symbol} = [:θ, :objective]
    )
    ξ = MvNormal(step_size)
    data = init_data_tuple(state, collect_data, n_steps)

    for i in 1:n_steps
        state.θ = state.θ - step_size/2 .* state.gradient .+ rand(ξ)
        state.gradient = gradient(state.θ)
        state.objective = objective(state.θ)
        state.counter += 1

        add_state!(data, state, i)
    end
    data = simplify_data(data)
    data, state
end


"""
As above, but the initial state is induced from init_θ, rather than explicitly
providing a startings state.

$(SIGNATURES)
"""
function langevin_diffusion(
    init_θ::AbstractVector{Float64};
    objective::Function,
    gradient::Function,
    step_size::Vector{Float64},
    n_steps::Integer,
    collect_data::Vector{Symbol} = [:θ, :objective]
    )
    state = GradientState(init_θ, objective(init_θ), gradient(init_θ))
    langevin_diffusion(
        state; objective, gradient, step_size, n_steps, collect_data
        )
end
