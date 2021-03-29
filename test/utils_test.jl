using SyntheticLikelihood, Test, Statistics,
    LinearAlgebra, Distributions, Random

## Test peturb
true_mean = [1.,1000]
true_cov = [9 0; 9 0]

d = MvNormal(2, 3) # Zero mean sd 5
peturbed = peturb(true_mean, d, 1000)

sample_mean = mean.(eachcol(peturbed))
sample_cov = cov(peturbed)

@test isapprox(true_mean, sample_mean; atol = 3)
@test isapprox(true_cov, sample_cov; atol = 20)  # Note uses norm

## Test stack_arrays
stack_arrays = SyntheticLikelihood.stack_arrays
VV = Vector([[1, 2], [3, 4], [5, 6]]) #VecVec
AV = Vector([[1 1; 1 1], [2 2; 2 2], [1 2; 3 4]])  # ArrayVec
VV
AV_expected = Array{Int64}(undef, (3,2,2))
AV_expected[1, :, :] = fill(1, 2, 2)
AV_expected[2, :, :] = fill(2, 2, 2)
AV_expected[3, :, :] = [1 2; 3 4]

@test stack_arrays(VV) == [1 2; 3 4; 5 6]
@test stack_arrays(AV) == AV_expected
@test_throws AssertionError stack_arrays([[1,2], [1,2,3]])


ensure_posdef = SyntheticLikelihood.ensure_posdef
seed = Random.seed!(1)
A = rand(seed, 3,3); A = -(A'*A)
A = Symmetric(A)

threshold = 1.
@test !isposdef(A)
A = ensure_posdef(A, threshold)

@test isposdef(A)
@test all(eigvals(A) .>= (threshold - 1e-10))
