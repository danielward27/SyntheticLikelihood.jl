# Contains methods to ensure hessian/covariance/correlation is positive definite
# and are reasonable.

abstract type AbstractRegularizer end


# TODO Update these docs
"""
Carries out the following steps:
1. Use `soft_abs` to get absolute values of the eigenvalues.
3. Decompose the covariance into the variance and correlation.
4. Divide the correlation matrix by a constant to reach condition criteria.
5. Set non diagonal correlation to zero if < τ
6. Scale so that the determinant matches a reference distribution.

See http://parker.ad.siu.edu/Olive/slch6.pdf

"""
@kwdef struct KitchenSink <: AbstractRegularizer
    "Reference/target log-determinant"
    target_logdet::Float64 = -23.
    "soft abs threshold. Minimum eigenvalue is 1/α. α=Inf is absolute value."
    α::Float64 = 10^6
    """Threshhold condition number of the correlation matrix."""
    c::Float64 = 10.
    "Threshold correlation below which set to zero."
    τ::Float64 = 0.1
end

function regularize(Σ::Union{Diagonal, Symmetric}, method::KitchenSink)
    @unpack target_logdet, α, c, τ = method
    summaries = [cond, det, min_var, max_var, mean_off_diag]
    logger = ObjectSummaryLogger(;summaries)  # For debugger
    add_log!(logger, "Initial P", Σ)

    Σ = soft_abs(Σ, α)
    add_log!(logger, "Post soft_abs", Σ)

    Σ = regularize_Σ_cor(Σ, c, τ)
    add_log!(logger, "Post regularize_Σ_cor", Σ)

    # Scale determinant to match ref
    Σ = cov_det_reg(Σ, target_logdet)
    add_log!(logger, "Post cov_det_reg", Σ)

    # # Limit variance
    # upper_σ²_lim = 100
    # lower_σ²_lim = 0.01

    # ref_σ² = diag(ref)
    # upper_σ² = upper_σ²_lim*ref_σ²
    # lower_σ² = lower_σ²_lim*ref_σ²
    # σ²[σ² .> upper_σ²] = upper_σ²[σ² .> upper_σ²]
    # σ²[σ² .< lower_σ²] = lower_σ²[σ² .< lower_σ²]

    # P = cor_to_cov(P, σ²)
    # add_log!(logger, "After modifying variances", P)

    @debug "$(get_pretty_table(logger))"

    Σ
end


"""
Takes "soft" absolute value of the eigenvalues, using the method of
Betancourt 2013 (https://arxiv.org/pdf/1212.4693.pdf).
α → Inf then this approaches the actual absolute value. Minimum eigenvalues
are limited to 1/α.
"""
function soft_abs(A::Union{Diagonal, Eigen, Symmetric}, α::Float64)
    @assert α > 0
    A = eigen(A)
    λ = A.values
    λ = @. λ*(exp(α*λ) + exp(-α*λ))/(exp(α*λ) - exp(-α*λ))
    λ[.!isfinite.(λ)] .= abs.(A.values[.!isfinite.(λ)])   # For when exp becomes Inf.
    A.values .= λ
    @assert all(isfinite.(Matrix(A)))
    Symmetric(Matrix(A))
end


"""
Regularize covariance matrix, by limiting condition number of the
associated correlation matrix, and setting off diagonal values of the
correlation below a threshold (τ) to zero.
"""
function regularize_Σ_cor(Σ::Union{Symmetric, Diagonal}, c::Float64, τ::Float64)
    @assert τ >=0 && τ < 1 && c > 0
    σ² = diag(Σ)
    R = cov_to_cor(Σ)
    λ = eigvals(R)
    cond = maximum(λ)/minimum(λ)
    if cond > c
        δ = maximum([0, (λ[end] - c*λ[1])/(c-1)])  # λ[1] is smallest eigval
        R = (1/(1+δ)).*(R + δ*I)
    end
    R = Matrix(R)
    R[R .< τ] .= 0
    R = Symmetric(R)
    Σ = cor_to_cov(R, σ²)
end



"""
Scale matrix to a particular log-determinant value.
"""
function cov_det_reg(
    Σ::Union{Symmetric, Diagonal},
    target_logdet::Float64
    )
    log_scale = 1/size(Σ, 1) * (target_logdet - logdet(Σ))
    Σ = exp(log_scale)*Σ
    Σ
end

# Summary functions for debugging/logging
max_var(P::AbstractMatrix) = maximum(diag(P))
min_var(P::AbstractMatrix) = minimum(diag(P))
mean_off_diag(M::AbstractMatrix) = begin
    non_diag = M[setdiff(1:length(M), diagind(M))]
    mean(non_diag)
end










#Use weighted average of determinant to scale P. α is the weight parameter,
#If α=0, then P is scaled to have determinant matching the reference, as
#α→∞, P is not scaled.
#"""
#function cov_det_reg(
#    P::AbstractMatrix,
#    ref_cov::AbstractMatrix,
#    α::Float64
#    )
#    @assert α >=0 && α <=1
#    det_P = det(P)
#    det_target = α*det_P + (1-α)*det(ref_cov)
#    scale = (det_target/det_P)^(1/size(P, 1))
#    P = scale * P
#    @assert det(P) ≈ det_target
#    P
#end
#







# # Just for now
# # TODO better method?
# """
# Uses the same as bove, but just inverts the result.
# """
# @kwdef struct KitchenSinkH <: AbstractRegularizer
#     "Thredhold condition number of the correlation matrix."
#     c = 50
#     "Threshold correlation below which set to zero."
#     τ = 0.05
#     "A reference distribution, e.g. a multiple of the prior or init_P."
#     ref::Sampleable
# end
#
# function regularize(H::Symmetric, method::KitchenSinkH)
#     @debug "- H Pre-reg H: cond=$(cond(H)) det=$(cond(H))"
#     @unpack c, τ, ref = method
#     ksp = KitchenSinkP(c, τ, ref)
#     H = regularize(H, ksp)^-1
#     @debug "- H Post-reg H: cond=$(cond(H)) det=$(cond(H))"
#     H
# end
