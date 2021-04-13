using SyntheticLikelihood, Test, LinearAlgebra

regularize_Σ_cor = SyntheticLikelihood.regularize_Σ_cor
cov_to_cor = SyntheticLikelihood.cov_to_cor
cor_to_cov = SyntheticLikelihood.cor_to_cov

A = rand(3,3); A = Symmetric(A'A)
σ² = diag(A)
R = cov_to_cor(A)

@test diag(R) ≈ ones(3)
@test regularize_Σ_cor(A, Inf, 1-1e-15) ≈ Diagonal(A)
@test regularize_Σ_cor(A, Inf, 0.) ≈ A

cov_det_reg = SyntheticLikelihood.cov_det_reg
@test logdet(cov_det_reg(A, 10.)) ≈ 10


soft_abs = SyntheticLikelihood.soft_abs
A = rand(3,3); A = Symmetric(A'A - I)
@test sort(abs.(eigvals(A))) ≈ eigvals(soft_abs(A, Inf))
