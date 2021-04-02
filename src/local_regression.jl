

"""
Struct for containing the objective function value, along with the
    gradient and hessian if appropriate (defualt to `nothing`).

    $(FIELDS)
"""
Base.@kwdef struct ObjGradHess
    objective::Float64
    gradient::Union{Vector{Float64}, Nothing} = nothing
    hessian::Union{Symmetric{Float64}, Nothing} = nothing
end


 """
 Likelihood objective, gradient and hessian estimation using local regressions.
 """
 function obj_grad_hess(local_likelihood::LocalLikelihood, θ::Vector{Float64})
      θᵢ = θ
      ll = local_likelihood
      θ = peturb(θᵢ, ll.P, ll.n_sim)
      s = simulate_n_s(θ; ll.simulator, ll.summary)
      s = remove_invariant(s)
      μ = quadratic_local_μ(; θᵢ, θ, s)
      Σ = glm_local_Σ(; θᵢ, θ, μ.ϵ)
      l = likelihood_obj_grad_hess(μ, Σ, ll.s_true)
      l.hessian .= ensure_posdef(l.hessian, ll.eigval_threshold)
      l
 end

# Note this does not ensure positive definiteness
 function likelihood_obj_grad_hess(
     μ::Localμ, Σ::LocalΣ,
     s_true::Vector{Float64})
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

  # Evaluate synthetic likelihood
  mvn = MvNormal(μ.μ, Σ.Σ)
  l = logpdf(mvn, sᵒ)

  return ObjGradHess(-l, -∂, -∂²)
end









"""
Get the objective, gradient and Hessian of negative log-posterior, from the prior and
the objective, gradient and hessian of the negative log-likelihood.

$(SIGNATURES)
"""
function posterior_obj_grad_hess(;
    prior::Sampleable,
    neg_likelihood_ogh::ObjGradHess,
    θ::Vector{Float64}
    )
    obj = neg_likelihood_ogh.objective - loglikelihood(prior, θ)
    grad = neg_likelihood_ogh.gradient - log_prior_gradient(prior, θ)
    hess = neg_likelihood_ogh.hessian - log_prior_hessian(prior, θ)
    ObjGradHess(obj, grad, hess)
end


"""
Estimate negative log-posterior, and its gradient and hessian. Uses local
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


function obj_grad_hess(
    local_posterior::LocalPosterior,
    θ::Vector{Float64}
    )

    neg_likelihood_ogh = obj_grad_hess(LocalLikelihood(local_posterior), θ)

    posterior_obj_grad_hess(;
        prior = local_posterior.prior, neg_likelihood_ogh, θ
        )
    # TODO Handle cases where prior support is bounded?
    # TODO need a way to check valid proposals in simulate n_s.
    # Maybe possible with Multiple dispatch with priors?
end










## Bayesian

# For Bayesian analyses we also need the gradient and hessian of -prior:

function log_prior_gradient(d::Sampleable, θ::Vector{Float64})
    f(θ) = loglikelihood(d, θ)
    ForwardDiff.gradient(f, θ)
end

function log_prior_hessian(d::Sampleable, θ::Vector{Float64})
    f(θ) = loglikelihood(d, θ)
    Symmetric(ForwardDiff.hessian(f, θ))
end
