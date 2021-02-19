

# Create noisy gaussian

# Toy example, noisy normal logpdf with means 1 and covariance 1
# Density plays role of summary statistic for local regression
# Provides simple sanity check between hessian and covariance matrix.
function noisy_normal_logpdf(θ::AbstractVector; μ::AbstractVector = [1,2,3])
    d = MvNormal(μ, 1)
    logpdf(d, θ) + rand()
end

θ_mle = [1,2,3]
θ = peturb(θ_mle, [0.5, 0.5, 0.5], 100)
s = noisy_normal_logpdf.(eachrow(θ))

# Now have θ and s. Regress θ on s:
θ = θ .- θ_mle'  # Demean

# Componenets
# Should make this a seperate poly function
# hcat?
# ones(size(θ)[1])
# θ
# θ.^2
# θ[:, 1]** θ [:, 2]

















# Use this knowledge https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s?rq=1
