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

# Outer constructor
function SamplerState(θ::AbstractVector{Float64}, ogh::ObjGradHess)
    SamplerState(θ, ogh.objective, ogh.gradient, ogh.hessian, 0)
end

function get_init_state(
    local_approximation::LocalApproximation,
    init_θ::Vector{Float64}
    )
    ogh = obj_grad_hess(local_approximation, init_θ)
    SamplerState(init_θ, ogh)
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
    "Step size parameter."
    step_size::Float64
end


function update!(sampler::Langevin, local_approximation, state::SamplerState)
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient
    ξ = rand(MvNormal(length(θ) , sqrt(η)))
    state.θ = θ .- η ./ 2 .* ∇ .+ ξ

    ogh = obj_grad_hess(local_approximation, state.θ)
    state.objective = ogh.objective
    state.gradient = ogh.gradient
    state.counter += 1
end



## Preconditioned Langevin diffusion
# TODO Change to match martin et al?
"""
Sampler object for Preconditioned Langevin diffusion. Also can be thought of as
    a stochastic newton method with constant step size. Uses the update:
    θ := θ - η/2 * H⁻¹*∇ + ξ, where ξ ∼ N(0, ηH⁻¹).

    $(FIELDS)
"""
mutable struct PreconditionedLangevin <: AbstractSampler
    "Step size"
    step_size::Float64
end


function update!(
    sampler::PreconditionedLangevin,
    local_approximation::LocalApproximation,
    state::SamplerState,
    )

    η, θ, ∇, H  = sampler.step_size, state.θ, state.gradient, state.hessian
    ξ = rand(MvNormalCanon(1/η .* H))  # Equiv to N(0, η.*H⁻¹)
    H = Symmetric(H)
    state.θ = θ .- η/2 .* (H \ ∇) .+ ξ

    ogh = obj_grad_hess(local_approximation, state.θ; )
    state.objective = ogh.objective
    state.gradient = ogh.gradient
    state.hessian = ogh.hessian
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
function run_sampler!(
    sampler::AbstractSampler,
    local_approximation::LocalApproximation;
    init_θ::Vector{Float64},
    n_steps::Integer,
    collect_data::Vector{Symbol} = [:θ, :objective])

    state = get_init_state(local_approximation, init_θ)
    data = init_data_tuple(state, collect_data, n_steps)

    for i in 1:n_steps
        update!(sampler, local_approximation, state)
        add_state!(data, state, i)
    end
    simplify_data(data)
end
