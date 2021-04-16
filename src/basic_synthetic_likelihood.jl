



function basic_synthetic_likelihood(
    local_approximation::LocalApproximation,
    θ::Vector{Float64}
    )
    @unpack simulator, summary, n_sim, s_true = local_approximation
    s = simulate_n_s(θ; simulator, summary, n_sim)
    s = rm_outliers(s)
    s, s_true = remove_invariant(s, s_true)
    Σ = cov(s)
    μ = mean.(eachcol(s))
    d = MvNormal(μ, Σ)
    logpdf(d, s_true)
end
