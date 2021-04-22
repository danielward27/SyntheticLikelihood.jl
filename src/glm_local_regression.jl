

"""
Struct that contains the estimated local properties of Σ (the covariance matrix
of the summary statistics).

$(FIELDS)
- Σ The (nₛ×nₛ) (estimated) covariance matrix of the summary statistics.
- ∂ The (nₛ×nₛ×n_θ) matrix of estimated first derivitives of Σ.
"""
struct LocalΣ
    Σ::Symmetric{Float64}
    ∂::Array{Float64, 3}

    function LocalΣ(Σ, ∂)
        @assert size(Σ) == size(∂)[1:2]
        new(Σ, ∂)
    end
end


"""
Use a gamma distributed GLM with log link function to estimate the local properties
    of the covariance matrix of the statistics Σ.  θ should not have a bias term (added internally).

Specifically, this function:
- Creates a rough initial Σ estimate using `cov(ϵ)`.
- Estimates the diagonal elements Σⱼⱼ, and ∂Σⱼⱼ using local regression.
- Esimates off-diagonal elements of Σ by scaling the sample correlation matrix
    with √Σⱼⱼ (standard deviations).
- Esimate off-diagonal gradients ∂Σᵢⱼ by averaging the coefficients associated
with indices i and j.

$(SIGNATURES)

# Arguments
- `θᵢ::AbstractVector`  Original θ (used for centering).
- `θ::AbstractMatrix` Peturbed θ from local area.
- `ϵ::AbstractMatrix` Residuals from quadratic regression (n_sim × n_sumstats).
"""
function glm_local_Σ(;
    θᵢ::AbstractVector,
    θ::AbstractMatrix,
    ϵ::AbstractMatrix)

    nₛ = size(ϵ, 2)
    n_θ = length(θᵢ)
    θ = θ .- θᵢ'  # Center
    θ = hcat(ones(size(θ, 1)), θ)  # Bias
    ϵ² = ϵ.^2  # Distributed as ϵ² ∼ exp(ϕ + ∑vₖθₖ)z, z ∼ χ²(1)

    samp_Σ = cov(ϵ)
    samp_Ψ, = cov_to_cor(Symmetric(samp_Σ))

    # Get coefficients of GLM
    coefs = Array{Float64}(undef, nₛ, n_θ+1)
    ϵ² = min.(ϵ², 15)  # Occasionally too high for exp
    @debug """
    Maximum residual = $(maximum(ϵ²))
    Median residual = $(median(ϵ²))
    """

    for i in 1:nₛ
        try
            fit = glm(θ, ϵ²[:, i], Gamma(), LogLink(), maxiter=1000) # TODO: Add weights?
            coefs[i, :] = coef(fit)
        catch e
            if e isa StatsBase.ConvergenceException
                @warn """
                GLM did not converge. Corresponding variance set to sample
                covariance.
                """
                fallback = zeros(n_θ)
                fallback[1] = log(samp_Σ[i,i])
                coefs[i, :] = fallback
            else
                throw(e)
            end

        end
    end

    ϕ = coefs[:, 1]  # exp(ϕ) gives variance estimates
    v = coefs[:, 2:end]

    # Estimate the gradients
    ∂ = Array{Float64}(undef, nₛ, nₛ, n_θ)
    for i in 1:nₛ for j in 1:nₛ  # upper traingular
        ∂[i, j, :] = ∂[j, i, :] =
         samp_Ψ[i, j] * exp((ϕ[i] + ϕ[j])/2) .* ((v[i, :] .+ v[j, :])/2)
    end end

    # Estimate covariance matrix
    sds = Diagonal(exp.(ϕ/2))
    Σ = sds * samp_Ψ * sds

    LocalΣ(Symmetric(Σ), ∂)
end
