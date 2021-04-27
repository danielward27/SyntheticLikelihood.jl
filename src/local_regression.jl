

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
 Negative log-likelihood objective, gradient and hessian estimation using local
 regressions. Note that this does not ensure positive definiteness of the
 Hessian.
 """
 function likelihood_obj_grad_hess(
     local_approximation::LocalApproximation,
     θ::Vector{Float64}
     )
     @unpack simulator, summary, P, n_sim, s_true, valid_params = local_approximation
     θᵢ = θ
     θ = peturb(θᵢ, P, valid_params; n = n_sim)
     s = simulate_n_s(θ; simulator, summary)
     s, θ = rm_outliers(s, θ)
     s, s_true = remove_invariant(s, s_true)
     s, s_true = standardize(s, s_true)
     μ = quadratic_local_μ(; θᵢ, θ, s)
     Σ = glm_local_Σ(; θᵢ, θ, μ.ϵ)
     l = likelihood_calc(μ, Σ, s_true)
     l
 end


"""
Use results from local regressions to estimate the negative log-likelihood
function value, gradient and hessian.
"""
 function likelihood_calc(
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
Get the posterior, gradient and Hessian of negative log-posterior, from the
prior and the objective, gradient and hessian of the negative log-likelihood.
"""
function posterior_calc(
    prior::Prior,
    neg_likelihood_ogh::ObjGradHess,
    θ::Vector{Float64}
    )
    obj = neg_likelihood_ogh.objective - logpdf(prior, θ)
    grad = neg_likelihood_ogh.gradient - log_prior_gradient(prior, θ)
    hess = neg_likelihood_ogh.hessian - log_prior_hessian(prior, θ)
    ObjGradHess(obj, grad, hess)
end


"""
Estimate negative log-likelihood, and its gradient and hessian.
"""
function obj_grad_hess(local_likelihood::LocalLikelihood, θ)
    likelihood_obj_grad_hess(local_likelihood, θ)
end


"""
Estimate negative log-posterior, and its gradient and hessian.
"""
function obj_grad_hess(local_posterior::LocalPosterior, θ::Vector{Float64})
    neg_likelihood_ogh = likelihood_obj_grad_hess(local_posterior, θ)
    posterior_calc(local_posterior.prior, neg_likelihood_ogh, θ)
end
