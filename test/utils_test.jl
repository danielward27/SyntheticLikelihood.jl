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

AV_expected = Array{Int64}(undef, (3,2,2))
AV_expected[1, :, :] = fill(1, 2, 2)
AV_expected[2, :, :] = fill(2, 2, 2)
AV_expected[3, :, :] = [1 2; 3 4]

@test stack_arrays(VV) == [1 2; 3 4; 5 6]
@test stack_arrays(AV) == AV_expected
@test_throws AssertionError stack_arrays([[1,2], [1,2,3]])


remove_invariant = SyntheticLikelihood.remove_invariant
@test remove_invariant([1 1 1; 2 1 2], [1,2,3]; warn=false) ==  ([1 1; 2 2], [1,3])

cov_to_cor = SyntheticLikelihood.cov_to_cor
A = rand(3,3); A = Symmetric(A'A + I)
σ² = diag(A)

R = cov_to_cor(A)
@test diag(R) ≈ ones(3)

cor_to_cov = SyntheticLikelihood.cor_to_cov
@test A ≈ cor_to_cov(R, σ²)

## Test the object summary logger
ObjectSummaryLogger = SyntheticLikelihood.ObjectSummaryLogger
add_log! = SyntheticLikelihood.add_log!
get_pretty_table = SyntheticLikelihood.get_pretty_table

logger = ObjectSummaryLogger(summaries = [cond, det])
A = diagm(ones(3))
add_log!(logger, "A summary", A)

B = diagm(fill(2,3))
add_log!(logger, "B summary", B)
@test logger.data == ["A summary" 1.0 1.0; "B summary" 1.0 8.0]


standardize = SyntheticLikelihood.standardize

X = rand(3,3)*100
X, _, _ = standardize(X)

expected_mean = fill(0, size(X, 2))
actual_mean = mean.(eachcol(X))
expected_sd = ones(size(X, 2))
actual_sd = std.(eachcol(X))

@test isapprox(expected_mean, actual_mean; atol=1e-10)
@test isapprox(expected_sd, actual_sd; atol=1e-10)

X = rand(3,3)*100
y = rand(3)*100
X, y = standardize(X, y)
actual_mean = mean.(eachcol(X))
actual_sd = std.(eachcol(X))

@test isapprox(expected_mean, actual_mean; atol=1e-15)
@test isapprox(expected_sd, actual_sd; atol=1e-15)
