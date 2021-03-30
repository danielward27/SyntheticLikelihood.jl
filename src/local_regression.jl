
abstract type LocalApproximation end


"""
Contains the hyperparameters for getting a local approximation
of the negative log-likelihood surface using local regressions.

$(FIELDS)
"""
Base.@kwdef struct LocalLikelihood <: LocalApproximation
    "The simulator function taking parameter vector θ."
    simulator::Function
    "The summary function taking output from the simulator."
    summary::Function=identity
    "The observed summary statistics."
    s_true::Vector{Float64}
    "Distribution used to peturb parameter value"
    P::Sampleable
    "The number of peturbed points to use for the local regression."
    n_sim::Integer
    """Minimum eigenvalue threshold for the estimated hessian. Negative
    eigenvalues are flipped and those smaller than the threshold are
    set to the threshold."""
    eigval_threshold::Float64 = 0.5
end




"""
Contains the hyperparameters for getting a local approximation
of the posterior (using `LocalLikelihood` and a prior).
"""
Base.@kwdef struct LocalPosterior <: LocalApproximation
    prior::Sampleable
    simulator::Function
    summary::Function=identity
    s_true::Vector{Float64}
    P::Sampleable
    n_sim::Integer
    eigval_threshold::Float64 = 0.5
end


 """
 Likelihood objective, gradient and hessian estimation using local regressions.
 """
 function likelihood_obj_grad_hess(
      local_likelihood::LocalLikelihood,
      θ::Vector{Float64}
      )
      θᵢ = θ
      ll = local_likelihood
      θ = peturb(θᵢ, ll.P, ll.n_sim)
      s = simulate_n_s(θ; ll.simulator, ll.summary)
      s = remove_invariant(s)
      μ = quadratic_local_μ(; θᵢ, θ, s)
      Σ = glm_local_Σ(; θᵢ, θ, μ.ϵ)
      l = likelihood_obj_grad_hess(μ, Σ, ll.s_true; ll.eigval_threshold)
      l
 end

 function likelihood_obj_grad_hess(
     μ::Localμ, Σ::LocalΣ,
     s_true::Vector{Float64};
     eigval_threshold::Float64)
   sᵒ = s_true
   n_θ = size(μ.∂, 2)
   ∂ = Vector{Float64}(undef, n_θ)
   ∂² = Matrix{Float64}(undef, n_θ, n_θ)

   qrΣ = qr(Σ.Σ)
   Σ⁻¹sᵒ₋μ = qrΣ \ (sᵒ - μ.μ)  # Precalculate Σ⁻¹(sᵒ-μ)

   # Gradient
   for k in 1:n_θ
     Σ⁻¹∂Σₖ = qrΣ \ Σ.∂[:, :, k]
     ∂[k] = (μ.∂[:,k]' * Σ⁻¹sᵒ₋μ +
         0.5*(sᵒ - μ.μ)' * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ - 0.5*tr(Σ⁻¹∂Σₖ))
     end

  # Hessian
  for k in 1:n_θ for l in k:n_θ  # Upper traingular of hessian matrix
    Σ⁻¹∂Σₗ = qrΣ \ Σ.∂[:, :, l]
    Σ⁻¹∂Σₖ = qrΣ \ Σ.∂[:, :, k]
    ∂²[k, l] = (
      μ.∂²[:, k, l]' * Σ⁻¹sᵒ₋μ - μ.∂[:, k]' * Σ⁻¹∂Σₗ * Σ⁻¹sᵒ₋μ - μ.∂[:, k]'* (qrΣ \ μ.∂[:, l])
      - μ.∂[:, l]' * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ - (sᵒ - μ.μ)' * Σ⁻¹∂Σₗ * Σ⁻¹∂Σₖ * Σ⁻¹sᵒ₋μ
      + (1/2)*tr(Σ⁻¹∂Σₗ * Σ⁻¹∂Σₖ)
      )
  end end
  ∂² = Symmetric(∂²)
  neg_∂² = ensure_posdef(-∂², eigval_threshold)

  # Evaluate likelihood
  mvn = MvNormal(μ.μ, Σ.Σ)
  l = logpdf(mvn, sᵒ)

  return ObjGradHess(-l, -∂, neg_∂²)
end




function obj_grad_hess(
    local_posterior::LocalPosterior,
    θ::Vector{Float64}
    )
    error("unimplemented")
end










## Bayesian

# For Bayesian analyses we also need the gradient and hessian of -prior:

function neg_prior_gradient(d::Sampleable, θ::Vector{Float64})
    f(θ) = -loglikelihood(d, θ)
    ForwardDiff.gradient(f, θ)
end

function neg_prior_hessian(d::Sampleable, θ::Vector{Float64})
    f(θ) = -loglikelihood(d, θ)
    ForwardDiff.hessian(f, θ)
end

"""
Estimate negative log-posterior, and its gradient and hessian. Uses a local
regressions to first estimate the gradient and hessian of the likelihood
function, using `local_likelihood`, then uses the chain rule to
calculate the gradient of the posterior.

Prior gradient and hessian are calculated with
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). Univariate priors
can be specified using a `Product` distribution from Distributions.jl.
Or a multivariate prior can be specified using a multivariate distribution
Distributions.jl.

$(SIGNATURES)

## Arguments
- `θ` Starting parameter values.
- `prior` A vector of distributions (from Distributions package). Note that
    the distributions can be univariate or multivariate, but the overall dimension
    must match that of the θ (and the order must be consistent).
- `kwargs...` Key word arguments passed to `local_likelihood`.
"""

function local_posterior(
    θ::Vector{Float64};
    prior::Vector{Sampleable},
    kwargs...
    )
    @assert length(θ) == sum([length(p) for p in prior])


    # TODO Handle cases where prior support is bounded?


    # Calculate likelihood

    # TODO need a way to check valid proposals below
    l =  local_likelihood(θ; kwargs...)




end
