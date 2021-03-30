#---- General type definitions ----

"""
Struct for containing the state of sampler at a particular iteration.
    Gradient and hessian are `nothing` unless specified.

    $(FIELDS)
"""
Base.@kwdef mutable struct SamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::Union{AbstractVector, Nothing} = nothing
    hessian::Union{AbstractMatrix, Nothing} = nothing
    counter::Integer = 0
end


"""
Struct for containing the objective function value, along with the
    gradient and hessian if appropriate (defualt to `nothing`).

    $(FIELDS)
"""
Base.@kwdef struct ObjGradHess
    objective::Float64
    gradient::Union{AbstractVector, Nothing} = nothing
    hessian::Union{AbstractMatrix, Nothing} = nothing
end


#------ Sampling algorithms -------
abstract type AbstractSampler end

"""
Sampler object for Langevin diffusion. Uses a discrete time Euler approximation of
the Langevin diffusion (unadjusted Langevin algorithm), given by the update
θ := θ - η/2 * ∇ + ξ, where ξ is given by N(0, ηI).

$(FIELDS)
"""
mutable struct Langevin <: AbstractSampler
    step_size::Float64
    "Step size parameter."
    obj_grad_hess::Function
    "Must return ObjGradHess object with objective and gradient fields."
    kwargs
    "kwargs to be passed to obj_grad_hess."
    Langevin(step_size, obj_grad_hess; kwargs...) =
        new(step_size, obj_grad_hess, kwargs)
end

function get_init_state(
    sampler::Langevin,
    init_θ::Vector{Float64},
    kwargs...)

    la = sampler.obj_grad_hess(init_θ; sampler.kwargs...)

    SamplerState(;θ = init_θ,
                objective = la.objective,
                gradient = la.gradient)
end

function update!(sampler::Langevin, state::SamplerState, kwargs...)
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient
    ξ = rand(MvNormal(length(θ) , sqrt(η)))
    state.θ = θ .- η ./ 2 .* ∇ .+ ξ

    la = sampler.obj_grad_hess(state.θ; sampler.kwargs...)
    state.objective = la.objective
    state.gradient = la.gradient
    state.counter += 1
end



## Preconditioned Langevin diffusion
"""
Sampler object for Preconditioned Langevin diffusion. Also can be thought of as
    a stochastic newton method. Uses the update:
    θ := θ - η/2 * H⁻¹*∇ + ξ, where ξ ∼ N(0, ηH⁻¹).

    $(FIELDS)
"""
mutable struct PreconditionedLangevin <: AbstractSampler
    "Step size"
    step_size::Float64
end


function get_init_state(
    sampler::PreconditionedLangevin,
    init_θ::Vector{Float64},
    kwargs...)

    la = sampler.obj_grad_hess(init_θ; sampler.kwargs...)

    SamplerState(;θ = init_θ,
                objective = la.objective,
                gradient = la.gradient,
                hessian = la.hessian)
end


function update!(
    sampler::PreconditionedLangevin,
    state::SamplerState,
    kwargs...
    )
    η, θ, ∇, H  = sampler.step_size, state.θ, state.gradient, state.hessian
    ξ = rand(MvNormalCanon(1/η .* H))  # Equiv to N(0, η.*H⁻¹)
    H = Symmetric(H)
    state.θ = θ .- η/2 .* (H \ ∇) .+ ξ

    la = sampler.obj_grad_hess(state.θ; sampler.kwargs...)
    state.objective = la.objective
    state.gradient = la.gradient
    state.hessian = la.hessian
    state.counter += 1
end


#---- Run the sampling algorithm ----

"""
Run the sampling algorithm. Data to collect at each iteration is specified by
`collect_data`, and should be a subset of
`[:θ, :objective, :gradient, :hessian, :counter]`.

$(SIGNATURES)

Returns a tuple, with keys matching `collect_data`.
"""
function run_sampler!(;
    sampler::AbstractSampler,
    local_approximation::LocalApproximation,
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
