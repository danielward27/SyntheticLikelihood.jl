#------ Sampling algorithms -------
abstract type AbstractSampler end
abstract type AbstractSamplerState end

"""
Sampler for unadjusted langevin algorithm. Uses a discrete time Euler approximation of
the langevin diffusion, given by the update θ := θ - η/2 * ∇ + ξ,
where ξ is given by N(0, ηI).

$(FIELDS)
"""
@kwdef struct ULA <: AbstractSampler
    step_size::Float64
end

@kwdef mutable struct ULAState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::Vector{Float64}
    counter::Integer = 0
end

function get_init_state(
    sampler::ULA,
    local_approximation::LocalApproximation,
    init_θ::Vector{Float64}
    )
    ogh = obj_grad_hess(local_approximation, init_θ)
    @unpack objective, gradient = ogh
    ULAState(; θ=init_θ, objective, gradient)
end

function update!(
    sampler::ULA,
    local_approximation::LocalApproximation,
    state::ULAState
    )
    η, θ, ∇  = sampler.step_size, state.θ, state.gradient
    ξ = rand(MvNormal(length(θ) , sqrt(η)))
    state.θ = θ .- η ./ 2 .* ∇ .+ ξ

    ogh = obj_grad_hess(local_approximation, state.θ)
    state.objective = ogh.objective
    state.gradient = ogh.gradient
    state.counter += 1
end


## Riemannian ULA diffusion
"""
Sampler object for Riemannian ULA.
Uses the update: θ := θ - ϵ²H⁻¹*∇ - ϵ√H⁻¹ z, where z ∼ N(0, I).

    $(FIELDS)
"""
@kwdef struct RiemannianULA <: AbstractSampler
    step_size::Float64
end

@kwdef mutable struct RiemannianULAState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    gradient::Vector{Float64}
    hessian::Symmetric{Float64}
    counter::Integer = 0
end

function get_init_state(
    sampler::RiemannianULA,
    local_approximation::LocalApproximation,
    init_θ::Vector{Float64}
    )
    ogh = obj_grad_hess(local_approximation, init_θ)
    @unpack objective, gradient, hessian = ogh
    RiemannianULAState(; θ=init_θ, objective, gradient, hessian)
end

function update!(
    sampler::RiemannianULA,
    local_approximation::LocalApproximation,
    state::RiemannianULAState
    )
    ϵ, θ, ∇  = sampler.step_size, state.θ, state.gradient
    @unpack gradient, θ, hessian = state

    la = local_approximation

    H⁻¹ = state.hessian^-1
    H⁻¹ = regularize(H⁻¹, la.P_regularizer)
    la.P = MvNormal(H⁻¹)

    z = randn(length(θ))
    state.θ = θ .- (ϵ^2 .* H⁻¹*∇ )/2 .- ϵ*sqrt(H⁻¹) * z

    ogh = obj_grad_hess(la, state.θ)
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
