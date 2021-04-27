using SyntheticLikelihood, Test, Distributions, ForwardDiff, LinearAlgebra
using SyntheticLikelihood: cut_at, logpdf, log_prior_gradient,
    log_prior_hessian, insupport, sample

v = [1,2,3,4]
@test cut_at(v, [1,3]) == [[1], [2,3], [4]]

# test logpdf
test_θ = rand(3)
prod = Product([Normal(1), Normal(2), Normal(3)])
mvn = MvNormal([1, 2, 3], [1., 1, 1])

prior_uni = Prior(prod.v)
prior_mv = Prior([mvn])

# Check density calculated right
expected_density = logpdf(prod, test_θ)
@assert expected_density ≈ logpdf(mvn, test_θ)

@test expected_density ≈ logpdf(prior_mv, test_θ)
@test expected_density ≈ logpdf(prior_uni, test_θ)

# Check gradient calculated right
expected_grad = gradlogpdf(mvn, test_θ)
@test expected_grad ≈ log_prior_gradient(prior_uni, test_θ)
@test expected_grad ≈ log_prior_gradient(prior_mv, test_θ)

# Check Hessian calculated right
expected_Hessian = begin
    f(θ) = loglikelihood(mvn, θ)
    Symmetric(ForwardDiff.hessian(f, test_θ))
end

@test expected_Hessian ≈ log_prior_hessian(prior_uni, test_θ)
@test expected_Hessian ≈ log_prior_hessian(prior_mv, test_θ)

prior_uni = Prior([Uniform(-5,5), Uniform(-10,10)])
@test insupport(prior_uni, [0., 0]) === true
@test insupport(prior_uni, [-6., 5]) === false
@test insupport(prior_uni, [-4., 11]) === false

prior_mv = Prior([MvLogNormal([1.,2.,3.])])
@test insupport(prior_mv, [1e7, 1e6, 1e5]) === true
@test insupport(prior_mv, [1e7, 1e6, -1e-10]) === false


prior = Prior([MvNormal([1.,2.]), Normal()])
@test sample(Prior) isa Vector{Float64}
@test sample(Prior, 10) isa Matrix{Float64}
@test size(sample(Prior, 10)) == (10,3)
