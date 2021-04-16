using SyntheticLikelihood, Test, LinearAlgebra

regularize_cor = SyntheticLikelihood.regularize_cor
cov_to_cor = SyntheticLikelihood.cov_to_cor
cor_to_cov = SyntheticLikelihood.cor_to_cov

A = rand(3,3); A = Symmetric(A'A)
R, σ² = cov_to_cor(A)

@test diag(R) ≈ ones(3)
@test regularize_cor(R, Inf, 1-1e-15) ≈ Diagonal(ones(3))
@test regularize_cor(R, Inf, 0.) ≈ R

cov_logdet_reg = SyntheticLikelihood.cov_logdet_reg
@test logdet(cov_logdet_reg(A, 10.)) ≈ 10


soft_abs = SyntheticLikelihood.soft_abs
A = rand(3,3); A = Symmetric(A'A - I)
@test sort(abs.(eigvals(A))) ≈ eigvals(soft_abs(A, Inf))


regularize_Σ_merge = SyntheticLikelihood.regularize_Σ_merge
A = Symmetric(fill(0.1, 10, 10))
ref = Diagonal(fill(10, 10))
result = regularize_Σ_merge(A, ref, 0.1, 2.)
@test diag(result) ≈ ones(10)

A = Symmetric(fill(3, 10, 10))
ref = Diagonal(fill(1, 10))
result = regularize_Σ_merge(A, ref, 0.1, 2.)
@test diag(result) == fill(2, 10)
