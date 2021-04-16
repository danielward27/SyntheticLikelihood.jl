# Contains methods to ensure hessian/covariance/correlation is positive definite
# and are reasonable.

abstract type AbstractRegularizer end

"""
Carries out the following steps:
1. Use `soft_abs` to get absolute values of the eigenvalues.
2. If required, shrink the variances towards the prior.
3. Decompose the covariance into the variance and correlation.
4. Divide the correlation matrix by a constant to reach condition criteria.
5. Set non diagonal correlation to zero if < τ

See http://parker.ad.siu.edu/Olive/slch6.pdf for correlation regularization.
"""
@kwdef struct KitchenSink <: AbstractRegularizer
    "Reference covariance matrix, e.g. prior or the initial P matrix."
    ref::Union{Diagonal, Symmetric}
    "soft abs threshold. Minimum eigenvalue is 1/α. α=Inf is absolute value."
    α::Float64 = 1 / (minimum(eigvals(ref))/1e3)
    """Minimum limit for variance compared to ref. Shrinks towards reference if
    exceeded. See `regularize_Σ_merge`."""
    var_lo::Float64 = 0.05
    "Maximum limit for variance compared to ref."
    var_hi::Float64 = 2
    """Maximum condition number of the associated correlation matrix. Shrinks
    correaltion towards the identity matrix if exceeded."""
    c::Float64 = 10.
    "Threshold correlation below which set to zero."
    τ::Float64 = 0.05
end

function regularize(Σ::Union{Diagonal, Symmetric}, method::KitchenSink)
    @unpack ref, α, var_lo, var_hi, c, τ = method
    summaries = [cond, det, min_var, max_var, min_off_diag, max_off_diag]
    logger = ObjectSummaryLogger(;summaries)  # For debugger
    add_log!(logger, "ref Σ", ref)
    add_log!(logger, "Initial Σ", Σ)

    Σ = soft_abs(Σ, α)
    add_log!(logger, "Post soft_abs Σ", Σ)

    Σ = regularize_Σ_merge(Σ, ref, var_lo, var_hi)
    add_log!(logger, "Post regularize_Σ_merge Σ", Σ)

    R, σ² = cov_to_cor(Σ)
    add_log!(logger, "Pre regularize_cor R", R)
    R = regularize_cor(R, c, τ)
    add_log!(logger, "Post regularize_cor R", R)

    Σ = cor_to_cov(R, σ²)
    add_log!(logger, "Final Σ", Σ)
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
Regularize correlation matrix, by limiting the condition number `c`, and
putting all correlations below threshold `τ` to zero.
"""
function regularize_cor(
    R::Union{Symmetric, Diagonal},
    c::Float64,
    τ::Float64
    )
    @assert c > 0
    @assert τ >= 0 && τ < 1
    λ = eigvals(R)
    cond = maximum(λ)/minimum(λ)
    if cond > c
        δ = maximum([0, (λ[end] - c*λ[1])/(c-1)])  # λ[1] is smallest eigval
        R = (1/(1+δ)).*(R + δ*I)
    end
    R = Matrix(R)
    R[abs.(R) .< τ] .= 0  # Don't need upper thresh. as we regularize cond.
    Symmetric(Matrix(R))
end


"""
Regularize the correlation matrix, by shrinking to the reference to make
variances fall within thresholds. lo and hi are multiplied by the variances
of the reference to find thresholds. Shrinkage is carried out using
Σ = αΣ + (1 - α)ref.
"""
function regularize_Σ_merge(
    Σ::Union{Diagonal, Symmetric},
    ref::Union{Diagonal, Symmetric},
    lo::Float64 = 0.1,
    hi::Float64 = 2.  # 2 Times the reference is limit
    )

    σ² = diag(Σ)
    ratios = σ² ./ diag(ref)
    if any(ratios .< lo)
        σ²ᵢ = σ²[findmin(ratios)[2]]
        r = diag(ref)[findmin(ratios)[2]]
        α = (r*(lo-1)) / (σ²ᵢ - r)
        Σ = α .* Σ + (1 - α) .* ref
    end

    if any(ratios .> hi)
        σ²ᵢ = σ²[findmax(ratios)[2]]
        r = diag(ref)[findmax(ratios)[2]]
        α = (r*(hi-1)) / (σ²ᵢ - r)
        Σ = α*Σ + (1 - α) * ref
    end
    Symmetric(Matrix(Σ))
end


"""
Scale matrix to a particular log-determinant value.
"""
function cov_logdet_reg(
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
min_off_diag(M::AbstractMatrix) = begin
    off_diag = M[setdiff(1:length(M), diagind(M))]
    minimum(off_diag)
end

max_off_diag(M::AbstractMatrix) = begin
    off_diag = M[setdiff(1:length(M), diagind(M))]
    maximum(off_diag)
end
