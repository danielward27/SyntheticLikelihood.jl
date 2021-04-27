"""
Plot the marginal priors and posterior, along with the true parameter values if
available.
$(SIGNATURES)
"""
function plot_prior_posterior_density(
    prior::Prior,
    θ_samples::AbstractMatrix;
    param_names = reshape(["θ$(i)" for i in 1:10], 1,10),
    )

    plots = Vector()
    for i in 1:size(θ_samples, 2)
      p = plot(prior.v[i], xlabel = param_names[i])
      density!(θ_samples[:, i], legend = false)
      push!(plots, p)
    end

    lab = ["prior" "posterior"]

    legend = plot((1:2)', framestyle = :none, label = lab, legend = :left)
    plot(plots..., legend)
end


function plot_prior_posterior_density(
    prior::Prior,
    θ_samples::AbstractMatrix,
    θ_true;
    θ_names = reshape(["θ$(i)" for i in 1:10], 1,10),
    )
    plots = Vector()
    for i in 1:size(θ_samples, 2)
        p = plot(prior.v[i], xlabel = θ_names[i])
        density!(θ_samples[:, i], legend = false)
        vline!([θ_true[i]])
        push!(plots, p)
    end
  
    label = ["prior" "posterior" "true θ"]
    legend = plot((1:3)', framestyle = :none, label = label, legend = :left)
    plot(plots..., legend)
  end