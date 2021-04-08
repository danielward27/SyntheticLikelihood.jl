# Contains methods to ensure hessian/covariance/correlation is positive definite
# and are reasonable.

abstract type AbstractRegularizer end

"""
Flip the eigenvalues.
"""
struct Flip <: AbstractRegularizer end

function regularize(H::Symmetric, method::Flip)
    H = eigen(H)
    H.values .= abs.(H.values)
    Symmetric(Matrix(H))
end

# TODO Update these docs
"""
Carries out the following steps:
1. Flip eigenvalues of hessian.
2. Invert hessian to get covariance matrix.
3. Decompose the covariance into the variance and correlation.
4. Divide the correlation matrix by a constant to reach condition criteria.
5. Set non diagonal correlation to zero if <τ
6. Scale so that the determinant matches a reference distribution.

See http://parker.ad.siu.edu/Olive/slch6.pdf

"""
@kwdef struct KitchenSink <: AbstractRegularizer
    "A reference covariance matrix, e.g. a multiple of init_P."
    ref::AbstractMatrix
    """Threshhold condition number of the correlation matrix."""
    c::Float64 = 10.
    "Threshold correlation below which set to zero."
    τ::Float64 = 0.1
end

function regularize(H::Symmetric, method::KitchenSink)
    @unpack ref, c, τ = method
    summaries = [cond, det, min_var, max_var, mean_off_diag]
    logger = ObjectSummaryLogger(;summaries)  # For debugger
    add_log!(logger, "Reference summary", ref)
    H = eigen(H)
    H.values .= abs.(H.values)
    P = Symmetric(H^-1)
    add_log!(logger, "Initial P", P)

    # Scale so determinant to matches ref
    P = cov_det_reg(P, ref)
    add_log!(logger, "After scaling determinant", P)

    # # Regularize correlation matrix
    σ² = diag(P)
    R = cov_to_cor(P)
    R = cor_cond_threshold(R, c)
    R[R .< τ] .= 0
    P = cor_to_cov(R, σ²)

    add_log!(logger, "After modifying correlation", P)

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

    P
end



# 6.6 in http://parker.ad.siu.edu/Olive/slch6.pdf
function cor_cond_threshold(R::Symmetric, c::Float64)
    λ = eigvals(R)
    δ = maximum([0, (λ[end] - c*λ[1])/(c-1)])  # λ[1] smallest eigval
    (1/(1+δ)).*(R + δ*I)
end


"""
Scale one matrix so the determinant matches another.
"""
function cov_det_reg(
    P::AbstractMatrix,
    ref::AbstractMatrix,
    )
    n = size(P, 1)
    log_scale = 1/n * (logdet(ref) - logdet(P))
    P = exp(log_scale)*P
    P
end



# Summary functions for debugging
max_var(P::AbstractMatrix) = maximum(diag(P))
min_var(P::AbstractMatrix) = minimum(diag(P))
function mean_off_diag(M::AbstractMatrix)
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
