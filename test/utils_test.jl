using SyntheticLikelihood, Test, Statistics, LinearAlgebra

## Test peturb
true_mean = [1.,1000]
true_cov = [50 10; 10 50]
peturbed = peturb(true_mean, true_cov, 1000)

sample_mean = mean.(eachcol(peturbed))
sample_cov = cov(peturbed)

# Specified with matrix
@test_throws AssertionError peturb([1,2], rand(3,3))
@test isapprox(true_mean, sample_mean; atol = 3)
@test isapprox(true_cov, sample_cov; atol = 20)

# Specified with vector
true_cov = [50, 10]
peturbed = peturb(true_mean, true_cov, 1000)

sample_mean = mean.(eachcol(peturbed))
sample_cov = cov(peturbed)

@test isapprox(true_mean, sample_mean; atol = 3)
@test isapprox(diagm(true_cov), sample_cov; atol = 20)

## Test pairwise combinations
pc = SyntheticLikelihood.pairwise_combinations
@test pc(1) == [1 1]
@test pc(3) == [1 1; 1 2; 1 3; 2 2; 2 3; 3 3]
