using SyntheticLikelihood, Test, LinearAlgebra

using SyntheticLikelihood: regularize_cor, cov_to_cor, cor_to_cov,
    cov_logdet_reg, soft_abs, regularize_Σ_merge

A = rand(3,3); A = Symmetric(A'A)
R, σ² = cov_to_cor(A)

@test diag(R) ≈ ones(3)
@test regularize_cor(R, Inf, 1-1e-15) ≈ Diagonal(ones(3))
@test regularize_cor(R, Inf, 0.) ≈ R
@test logdet(cov_logdet_reg(A, 10.)) ≈ 10

A = rand(3,3); A = Symmetric(A'A - I)
@test sort(abs.(eigvals(A))) ≈ eigvals(soft_abs(A, Inf))


A = Symmetric(fill(0.1, 10, 10))
ref = Diagonal(fill(10, 10))
result = regularize_Σ_merge(A, ref, 0.1, 2.)
@test diag(result) ≈ ones(10)

A = Symmetric(fill(3, 10, 10))
ref = Diagonal(fill(1, 10))
result = regularize_Σ_merge(A, ref, 0.1, 2.)
@test diag(result) == fill(2, 10)
