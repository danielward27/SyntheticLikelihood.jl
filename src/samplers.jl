#------ Sampling algorithms -------
abstract type AbstractSampler end
abstract type AbstractSamplerState end

"""
Sampler for Langevin diffusion. Uses a discrete time Euler approximation of
the Langevin diffusion (unadjusted Langevin algorithm), given by the update
θ := θ - η/2 * ∇ + ξ, where ξ is given by N(0, ηI).

$(FIELDS)
"""
@kwdef struct Langevin <: AbstractSampler
    step_size::Float64
end

@kwdef mutable struct LangevinState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::Vector{Float64}
    counter::Integer = 0
end

function get_init_state(
    sampler::Langevin,
    local_approximation::LocalApproximation,
    init_θ::Vector{Float64}
    )
    ogh = obj_grad_hess(local_approximation, init_θ)
    @unpack objective, gradient = ogh
    LangevinState(; θ=init_θ, objective, gradient)
end

function update!(
    sampler::Langevin,
    local_approximation::LocalApproximation,
    state::LangevinState
    )
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient
    ξ = rand(MvNormal(length(θ) , sqrt(η)))
    state.θ = θ .- η ./ 2 .* ∇ .+ ξ

    ogh = obj_grad_hess(local_approximation, state.θ)
    state.objective = ogh.objective
    state.gradient = ogh.gradient
    state.counter += 1
end



## Preconditioned Langevin diffusion
# TODO Change to match martin et al? Or to match the Reimannian one probably best.
"""
Sampler object for Preconditioned Langevin diffusion. Also can be thought of as
    a stochastic newton method with constant step size. Uses the update:
    θ := θ - η/2 * H⁻¹*∇ + ξ, where ξ ∼ N(0, ηH⁻¹).

    $(FIELDS)
"""
@kwdef struct PreconditionedLangevin <: AbstractSampler
    step_size::Float64
end

@kwdef mutable struct PreconditionedLangevinState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::Vector{Float64}
    hessian::Symmetric{Float64}
    counter::Integer = 0
end

function get_init_state(
    sampler::PreconditionedLangevin,
    local_approximation::LocalApproximation,
    init_θ::Vector{Float64}
    )
    ogh = obj_grad_hess(local_approximation, init_θ)
    @unpack objective, gradient, hessian = ogh
    PreconditionedLangevinState(; θ=init_θ, objective, gradient, hessian)
end

function update!(
    sampler::PreconditionedLangevin,
    local_approximation::LocalApproximation,
    state::PreconditionedLangevinState
    )
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient

    la = local_approximation
    P = regularize(state.hessian, la.P_regularizer)
    la.P = MvNormal(P)

    # TODO seperate H regularizer (e.g. sampler.H_regularizer) ?
    H⁻¹ = P

    ξ = rand(MvNormal(η .* H⁻¹))
    state.θ = θ .- η/2 .* H⁻¹*∇ .+ ξ

    ogh = obj_grad_hess(local_approximation, state.θ)
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
    collect_data::Vector{Symbol} = [:θ, :objective]
    )
    local_approximation.P = local_approximation.init_P

    state = get_init_state(sampler, local_approximation, init_θ)
    data = init_data_tuple(state, collect_data, n_steps)

    for i in 1:n_steps
        update!(sampler, local_approximation, state)
        add_state!(data, state, i)
    end
    simplify_data(data)
end



# P_from_objective_hessian
# Note this is the same as Σ^-1
