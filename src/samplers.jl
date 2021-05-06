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


## Riemannian Unadjusted Langevin Algorithm
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
    @unpack P_regularizer, valid_params = local_approximation

    la = local_approximation

    H⁻¹ = state.hessian^-1
    H⁻¹ = regularize(H⁻¹, P_regularizer)
    la.P = MvNormal(0.5*H⁻¹)

    z = randn(length(θ))

    Δ = (ϵ^2 .* H⁻¹*∇ )/2 .+ ϵ*sqrt(H⁻¹) * z
    Δ = halve_update_until_valid(Δ, θ, valid_params)

    state.θ = θ .- Δ
    ogh = obj_grad_hess(la, state.θ)
    state.objective = ogh.objective
    state.gradient = ogh.gradient
    state.hessian = 0.5*(state.hessian + ogh.hessian)  # average with last iteration
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
    collect_data::Vector{Symbol} = [:θ, :objective],
    progress = true
    )

    state = get_init_state(sampler, local_approximation, init_θ)
    data = init_data_tuple(state, collect_data, n_steps)

    if progress
        @showprogress for i in 1:n_steps
            @debug "Iteration $(i)" state.θ
            update!(sampler, local_approximation, state)
            add_state!(data, state, i)
        end
    else
        for i in 1:n_steps
            @debug "Iteration $(i)" state.θ
            update!(sampler, local_approximation, state)
            add_state!(data, state, i)
        end
    end
    simplify_data(data)
end


"""
Random walk metropolis algorithm. Uses the objective function
(and not gradient/Hessian information).

Note that this stores the θ values at each iteration, even if rejection occurs.
To get only the accepted values, provide collect_data = [:θ, :accepted] to 
`run_sampler!`, then filter the data, e.g. θ = data.θ[data.accepted, :].

$(SIGNATURES)
"""
@kwdef struct RWMetropolis <: AbstractSampler
    "The proposal distribution."
    q::Sampleable
end

@kwdef mutable struct RWMetropolisState <: AbstractSamplerState
    θ::AbstractVector{Float64}
    objective::Float64
    counter::Integer = 0
    accepted::Bool = 0
end

function get_init_state(
    sampler::RWMetropolis,
    local_approximation::LocalApproximation,
    init_θ::Vector{Float64}
    )
    ogh = obj_grad_hess(local_approximation, init_θ)
    @unpack objective = ogh
    RWMetropolisState(; θ=init_θ, objective)
end

function update!(
    sampler::RWMetropolis,
    local_approximation::LocalApproximation,
    state::RWMetropolisState
    )
    @unpack q = sampler
    @unpack θ, objective = state
    @unpack valid_params = local_approximation

    Δ = rand(q)

    θ′ = θ + Δ

    if !valid_params(θ′)
        state.accepted = false # don't need to evaluate likelihood
    else
        obj′ = obj_grad_hess(local_approximation, θ′).objective

        accept_probability = min(1, exp(obj′ - objective))

        if accept_probability > rand()
            state.objective = obj′
            state.θ = θ′
            state.accepted = true
        else
            state.accepted = false
        end
    end

    state.counter += 1
end


"""
Halve update term until valid proposal found (e.g. with non-zero prior density).
Returns the modified update term.

$(SIGNATURES)
## Arguments
- `θ` Current parameter vector.
- `Δ` Proposed update term.
- `valid_params` Returns true if valid and false if invalid proposal.
"""
function halve_update_until_valid(
    Δ::Vector{Float64},
    θ::Vector{Float64},
    valid_params::Function
    )
    for i in 1:100
        proposal = θ .- Δ
        if valid_params(proposal)
            return Δ
        else
            Δ = Δ ./ 2
        end
    end
    error("The sampler could not find a valid update.")
    nothing
end