using SyntheticLikelihood, Test, LinearAlgebra

cor_cond_threshold = SyntheticLikelihood.cor_cond_threshold
cov_to_cor = SyntheticLikelihood.cov_to_cor
cor_to_cov = SyntheticLikelihood.cor_to_cov

A = rand(3,3); A = A'A
σ² = diag(A)
R = cov_to_cor(A)
@test cond(cor_cond_threshold(R, 2.)) ≈ 2
@test cor_cond_threshold(R, 1e100) ≈ R


cov_det_reg = SyntheticLikelihood.cov_det_reg
ref = 2*A
@test det(ref) ≈ det(cov_det_reg(A, ref))


# Regularizing cor and scaling up shouldn't change variance
σ² = fill(10,10)
Σ = diagm(σ²)
R = cov_to_cor(Σ)
Σ2 = cor_to_cov(R, σ²)
@test  σ² ≈ diag(Σ2)
