

# Create noisy gaussian

# Toy example, noisy normal logpdf with means 1 and covariance 1
function noisy_normal_logpdf(θ::AbstractVector; μ::AbstractVector = [1,2,3])
    d = MvNormal(μ, 1)
    logpdf(d, θ) + rand()
end

θ_mle = [1,2,3]

θ_peturbed = peturb(θ_mle, [1.,1,1])

# θ_peturbed = peturb(θ_mle, I)





# Use this knowledge https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s?rq=1
# peturb([1,1,1], )




noisy_normal_logpdf([1.,1,1])
