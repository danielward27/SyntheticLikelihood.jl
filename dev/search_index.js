var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SyntheticLikelihood","category":"page"},{"location":"#SyntheticLikelihood","page":"Home","title":"SyntheticLikelihood","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SyntheticLikelihood package. The package is currently a work in progress.","category":"page"},{"location":"documentation/#Documentation","page":"Documentation","title":"Documentation","text":"","category":"section"},{"location":"documentation/#Sampling","page":"Documentation","title":"Sampling","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Private = false\nModules = [SyntheticLikelihood]\nPages = [\"samplers.jl\"]","category":"page"},{"location":"documentation/#SyntheticLikelihood.RiemannianULA","page":"Documentation","title":"SyntheticLikelihood.RiemannianULA","text":"Sampler object for Riemannian ULA. Uses the update: θ := θ - ϵ²H⁻¹*∇ - ϵ√H⁻¹ z, where z ∼ N(0, I).\n\nstep_size\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.ULA","page":"Documentation","title":"SyntheticLikelihood.ULA","text":"Sampler for unadjusted langevin algorithm. Uses a discrete time Euler approximation of the langevin diffusion, given by the update θ := θ - η/2 * ∇ + ξ, where ξ is given by N(0, ηI).\n\nstep_size\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.run_sampler!-Tuple{SyntheticLikelihood.AbstractSampler, LocalApproximation}","page":"Documentation","title":"SyntheticLikelihood.run_sampler!","text":"Run the sampling algorithm. Data to collect at each iteration is specified by collect_data, and should be a subset of [:θ, :objective, :gradient, :hessian, :counter].\n\nrun_sampler!(sampler, local_approximation; init_θ, n_steps, collect_data)\n\n\nReturns a tuple, with keys matching collect_data.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#Implementation-details","page":"Documentation","title":"Implementation details","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"To implement a new sampler:","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Create a subtype of AbstractSampler, which takes the hyperparameters.\nCreate a subtype of AbstractSamplerState which stores the state of the sampler   at each iteration.\nImplement the get_init_state and update! functions to update the state","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"(e.g. parameters, gradients, objective function value).","category":"page"},{"location":"documentation/#Objectives","page":"Documentation","title":"Objectives","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Private = false\nModules = [SyntheticLikelihood]\nPages = [\"local_approximation_structs.jl\"]","category":"page"},{"location":"documentation/#SyntheticLikelihood.LocalLikelihood","page":"Documentation","title":"SyntheticLikelihood.LocalLikelihood","text":"Contains the hyperparameters for getting a local approximation of the negative log-likelihood surface using local regressions.\n\nsimulator\nsummary\ns_true\nP\nInitial distribution used to peturb the parameter value.\nn_sim\nThe number of peturbed points to use for the local regression.\nP_regularizer\nMethod to regularise the inverse Hessian to a reasonable covariance matrix.\nvalid_params\nParameter constraints. Function that takes a parameter vector and returns true (for valid parameters) or false (for invalid parameters). Defualt is θ -> true\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.LocalPosterior","page":"Documentation","title":"SyntheticLikelihood.LocalPosterior","text":"Contains the hyperparameters for getting a local approximation of the posterior. In contrast to the likelihood version, a prior is provided, P is by defualt a MvNormal with covariance 0.5*cov(prior), and  valid_params checks whether proposed points fall within the prior support.\n\nprior\nPrior distribution (either multivariate or Product distribution)\nsimulator\nsummary\ns_true\nP\nn_sim\nP_regularizer\nvalid_params\n\n\n\n\n\n","category":"type"},{"location":"documentation/#Implementation-details-2","page":"Documentation","title":"Implementation details","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"To implement a new objective:","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Create a subtype of LocalApproximation defining the hyperpameters.\nImplement obj_grad_hess function to estimate the objective, gradient and   Hessian.","category":"page"},{"location":"documentation/#Matrix-regularisation","page":"Documentation","title":"Matrix regularisation","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"As the Hessian inferred from local regressions is not guarenteed to be positive definite, we may need to modify the Hessian (or its inverse) in some way.","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Private = false\nModules = [SyntheticLikelihood]\nPages = [\"matrix_regularizers.jl\"]","category":"page"},{"location":"documentation/#SyntheticLikelihood.KitchenSink","page":"Documentation","title":"SyntheticLikelihood.KitchenSink","text":"Carries out the following steps:\n\nUse soft_abs to get absolute values of the eigenvalues.\nIf required, shrink the variances towards the reference.\nDecompose the covariance into the variance and correlation.\nDivide the correlation matrix by a constant (if reqiured) to reach condition threshold.\nSet non diagonal correlation to zero if < τ\nReconstruct the covariance matrix.\n\nSee http://parker.ad.siu.edu/Olive/slch6.pdf for correlation regularization, and https://doi.org/10.1007/978-3-642-40020-9_35 for soft abs.\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.regularize-Tuple{Union{LinearAlgebra.Diagonal, LinearAlgebra.Symmetric}, KitchenSink}","page":"Documentation","title":"SyntheticLikelihood.regularize","text":"Regularise the covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#Plotting","page":"Documentation","title":"Plotting","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Some convenience functions for plotting:","category":"page"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Private = false\nModules = [SyntheticLikelihood]\nPages = [\"plotting.jl\"]","category":"page"},{"location":"documentation/#SyntheticLikelihood.plot_prior_posterior_density-Tuple{Distributions.Product, AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.plot_prior_posterior_density","text":"Plot the marginal priors and posterior, along with the true parameter values if available.\n\nplot_prior_posterior_density(prior, θ_samples; param_names)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#Private/Unexported-interface","page":"Documentation","title":"Private/Unexported interface","text":"","category":"section"},{"location":"documentation/","page":"Documentation","title":"Documentation","text":"Public = false\nModules = [SyntheticLikelihood]","category":"page"},{"location":"documentation/#SyntheticLikelihood.LocalΣ","page":"Documentation","title":"SyntheticLikelihood.LocalΣ","text":"Struct that contains the estimated local properties of Σ (the covariance matrix of the summary statistics).\n\nΣ\n∂\n\nΣ The (nₛ×nₛ) (estimated) covariance matrix of the summary statistics.\n∂ The (nₛ×nₛ×n_θ) matrix of estimated first derivitives of Σ.\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.Localμ","page":"Documentation","title":"SyntheticLikelihood.Localμ","text":"Struct that contains the estimated local properties of μ (the expected values of the summary statistics).\n\nFields\n\nμ::Float64 Means of the summary statistics.\n∂::Vector{Float64} First derivitives w.r.t. parameters (nₛ×n_θ).\n∂²::Matrix{Float64} Second derivitive w.r.t. parameters (nₛ×nθ×nθ).\nϵ::Vector{Float64} Residuals of predicted summary statistics (nₛ×nₛᵢₘ).\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.ObjGradHess","page":"Documentation","title":"SyntheticLikelihood.ObjGradHess","text":"Struct for containing the objective function value, along with the     gradient and hessian if appropriate (defualt to nothing).\n\nobjective\ngradient\nhessian\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.ObjectSummaryLogger","page":"Documentation","title":"SyntheticLikelihood.ObjectSummaryLogger","text":"Used to store and print information about an object. Useful for printing neat debugging messages.\n\n\n\n\n\n","category":"type"},{"location":"documentation/#SyntheticLikelihood.add_state!-Tuple{NamedTuple, SyntheticLikelihood.AbstractSamplerState, Integer}","page":"Documentation","title":"SyntheticLikelihood.add_state!","text":"Add data to the data tuple.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.analytic_mvn_posterior-Tuple{Distributions.AbstractMvNormal, Distributions.AbstractMvNormal}","page":"Documentation","title":"SyntheticLikelihood.analytic_mvn_posterior","text":"Gets the analytic posterior distribution from a normal prior and normal likelihood. Useful for testing.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.cor_to_cov-Tuple{AbstractMatrix{T} where T, AbstractVector{T} where T}","page":"Documentation","title":"SyntheticLikelihood.cor_to_cov","text":"Convert correlation matrix to covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.cov_logdet_reg-Tuple{Union{LinearAlgebra.Diagonal, LinearAlgebra.Symmetric}, Float64}","page":"Documentation","title":"SyntheticLikelihood.cov_logdet_reg","text":"Scale matrix to a particular log-determinant value.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.cov_to_cor-Tuple{Union{LinearAlgebra.Diagonal, LinearAlgebra.Symmetric}}","page":"Documentation","title":"SyntheticLikelihood.cov_to_cor","text":"Convert covariance matrix to correlation matrix. Returns tuple (R, σ²)\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.deterministic_test_simulator-Tuple{AbstractVector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.deterministic_test_simulator","text":"Deterministic simulator useful for testing.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.glm_local_Σ-Tuple{}","page":"Documentation","title":"SyntheticLikelihood.glm_local_Σ","text":"Use a gamma distributed GLM with log link function to estimate the local properties     of the covariance matrix of the statistics Σ.  θ should not have a bias term (added internally).\n\nSpecifically, this function:\n\nCreates a rough initial Σ estimate using cov(ϵ).\nEstimates the diagonal elements Σⱼⱼ, and ∂Σⱼⱼ using local regression.\nEsimates off-diagonal elements of Σ by scaling the sample correlation matrix   with √Σⱼⱼ (standard deviations).\nEsimate off-diagonal gradients ∂Σᵢⱼ by averaging the coefficients associated\n\nwith indices i and j.\n\nglm_local_Σ(; θᵢ, θ, ϵ)\n\n\nArguments\n\nθᵢ::AbstractVector  Original θ (used for centering).\nθ::AbstractMatrix Peturbed θ from local area.\nϵ::AbstractMatrix Residuals from quadratic regression (nsim × nsumstats).\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.halve_update_until_valid-Tuple{Vector{Float64}, Vector{Float64}, Function}","page":"Documentation","title":"SyntheticLikelihood.halve_update_until_valid","text":"Halve update term until valid proposal found (e.g. with non-zero prior density). Returns the modified update term.\n\nhalve_update_until_valid(Δ, θ, valid_params)\n\n\nArguments\n\nθ Current parameter vector.\nΔ Proposed update term.\nvalid_params Returns true if valid and false if invalid proposal.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.init_data_tuple-Tuple{SyntheticLikelihood.AbstractSamplerState, Vector{Symbol}, Integer}","page":"Documentation","title":"SyntheticLikelihood.init_data_tuple","text":"Initialises a named tuple containing Vectors with undefined values. Used with samplers to store results. State just provides an \"example\" state from which to infer types of vectors in the array. Names of the named tuple are the symbols provided.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.likelihood_calc-Tuple{SyntheticLikelihood.Localμ, SyntheticLikelihood.LocalΣ, Vector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.likelihood_calc","text":"Use results from local regressions to estimate the negative log-likelihood function value, gradient and hessian.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.likelihood_obj_grad_hess-Tuple{LocalApproximation, Vector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.likelihood_obj_grad_hess","text":"Negative log-likelihood objective, gradient and hessian estimation using local regressions. Note that this does not ensure positive definiteness of the Hessian.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.linear_regression-Tuple{AbstractMatrix{T} where T, AbstractVector{T} where T}","page":"Documentation","title":"SyntheticLikelihood.linear_regression","text":"Carry out linear regression. X should have a bias column. Returns tuple (β, ŷ).\n\nlinear_regression(X, y)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.log_prior_gradient-Tuple{Distributions.Sampleable, Vector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.log_prior_gradient","text":"Automatic differentiation to get prior gradient.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.log_prior_hessian-Tuple{Distributions.Sampleable, Vector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.log_prior_hessian","text":"Automatic differentiation to get prior Hessian.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.obj_grad_hess-Tuple{LocalLikelihood, Any}","page":"Documentation","title":"SyntheticLikelihood.obj_grad_hess","text":"Estimate negative log-likelihood, and its gradient and hessian.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.obj_grad_hess-Tuple{LocalPosterior, Vector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.obj_grad_hess","text":"Estimate negative log-posterior, and its gradient and hessian.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.outlier_rows-Tuple{AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.outlier_rows","text":"Find outlier rows in matrix, using threshold multiples of iqr above and below the median for each column.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.peturb-Tuple{AbstractVector{T} where T, Distributions.Sampleable}","page":"Documentation","title":"SyntheticLikelihood.peturb","text":"Peturb a vector using a user specified distribution (often MVN zero mean). Returns array of size (n, length(θ)). If  a prior is provided, proposals are checked to have prior support using insupport and are resampled if not.\n\npeturb(θ, d; n)\n\n\nArguments\n\nθ Parameter to peturb.\nd Distribution from which to sample (see Distributions.jl).\nn = 1 Number of peturbed vectors to return.\nvalid_params Return true if θ vector is valid, and false if invalid.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.posterior_calc-Tuple{Distributions.Sampleable, SyntheticLikelihood.ObjGradHess, Vector{Float64}}","page":"Documentation","title":"SyntheticLikelihood.posterior_calc","text":"Get the posterior, gradient and Hessian of negative log-posterior, from the prior and the objective, gradient and hessian of the negative log-likelihood.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.quadratic_design_matrix-Tuple{AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.quadratic_design_matrix","text":"Design matrix for quadratic regression. Bias term appended as first column internally. Returns a tuple, with the matrix and the corresponding indices multiplied, that give rise to each column. Note, indices [1, 1] corresponds to the bias term (so indices compared to original matrix is shifted).\n\nquadratic_design_matrix(X)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.quadratic_local_μ-Tuple{}","page":"Documentation","title":"SyntheticLikelihood.quadratic_local_μ","text":"Finds the local behaviour of the summary statistic mean μ. Uses quadratic linear regression to approximate the mean, gradient and hessian around θᵢ. Returns a Localμ struct (see above).\n\nquadratic_local_μ(; θᵢ, θ, s)\n\n\nArguments\n\nθᵢ::AbstractVector Original θ.\nθ::AbstractMatrix Peturbed θ (sampled from local area).\ns::AbstractMatrix Corresponding summary statistics to θ.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.regularize_cor-Tuple{Union{LinearAlgebra.Diagonal, LinearAlgebra.Symmetric}, Float64, Float64}","page":"Documentation","title":"SyntheticLikelihood.regularize_cor","text":"Regularize correlation matrix, by limiting the condition number c, and putting all correlations below threshold τ to zero.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.regularize_Σ_merge","page":"Documentation","title":"SyntheticLikelihood.regularize_Σ_merge","text":"Regularize the correlation matrix, by shrinking to the reference to make variances fall within thresholds. lo and hi are multiplied by the variances of the reference to find thresholds. Shrinkage is carried out using Σ = αΣ + (1 - α)ref.\n\n\n\n\n\n","category":"function"},{"location":"documentation/#SyntheticLikelihood.remove_invariant-Tuple{Any, Any}","page":"Documentation","title":"SyntheticLikelihood.remove_invariant","text":"remove columns that have zero variance\n\nremove_invariant(s, s_true)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.rm_outliers-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.rm_outliers","text":"Remove outlier rows from both matrices, using first matrix to determine outliers.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.rm_outliers-Tuple{AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.rm_outliers","text":"Remove outlier rows from matrix.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.simplify_data-Tuple{NamedTuple}","page":"Documentation","title":"SyntheticLikelihood.simplify_data","text":"Loop through named tuple and call stack_arrays on any vector whose elements are an array. Used at end of samplers.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.simulate_n_s-Tuple{AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.simulate_n_s","text":"As for above, but a Matrix of parameter values are used, carrying out one     simulation from each row of θ (and hence n_sim is not required).\n\nsimulate_n_s(θ; simulator, summary, parallel)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.simulate_n_s-Tuple{AbstractVector{T} where T}","page":"Documentation","title":"SyntheticLikelihood.simulate_n_s","text":"Simulates summary statistics from the model under a fixed parameter vector. n_sim is specified as the number of simulations. Simulations can be run on multiple threads using parallel = false. By defualt no summary statistic function is used (by passing the identity function).\n\nsimulate_n_s(θ; simulator, summary, n_sim, parallel)\n\n\nArguments\n\nθ::AbstractVector Parameter vector passed to simulator.\nsimulator::Function Simulator.\nsummary::Function Summary function that takes output of simulator (defualt identity).\nn_sim::Integer Number of simulations.\nparallel::Bool = false Whether to run on multiple threads.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.soft_abs-Tuple{Union{LinearAlgebra.Diagonal, LinearAlgebra.Eigen, LinearAlgebra.Symmetric}, Float64}","page":"Documentation","title":"SyntheticLikelihood.soft_abs","text":"Takes \"soft\" absolute value of the eigenvalues, using the method of Betancourt 2013 (https://arxiv.org/pdf/1212.4693.pdf). α → Inf then this approaches the actual absolute value. Minimum eigenvalues are limited to 1/α.\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.stack_arrays-Tuple{Vector{T} where T}","page":"Documentation","title":"SyntheticLikelihood.stack_arrays","text":"Stacks a vector of consitently sized arrays to make a new array with dimensions (length(x), dim(x[1])...).\n\nstack_arrays(x)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.standardize-Tuple{AbstractMatrix{T} where T, AbstractVector{T} where T}","page":"Documentation","title":"SyntheticLikelihood.standardize","text":"Standardize matrix and vector, using the mean and standard deviation of the matrix.\n\nstandardize(X, y)\n\n\n\n\n\n\n","category":"method"},{"location":"documentation/#SyntheticLikelihood.standardize-Tuple{AbstractMatrix{T} where T}","page":"Documentation","title":"SyntheticLikelihood.standardize","text":"Standardize to zero mean and standard deviation 1. Can also provide a vector which will be standardized using the mean and standard deviation of the matrix.\n\nstandardize(X)\n\n\n\n\n\n\n","category":"method"},{"location":"example/","page":"Example","title":"Example","text":"EditURL = \"<unknown>/example.jl\"","category":"page"},{"location":"example/#Example","page":"Example","title":"Example","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"To see how things work, its easiest to consider an example. Here we consider the simulating from the noisily observed ricker map.","category":"page"},{"location":"example/#Imports","page":"Example","title":"Imports","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"using SyntheticLikelihood, Distributions, StatsPlots, StatsBase, Random\nRandom.seed!(3)\nnothing #hide","category":"page"},{"location":"example/#Define-the-simulator","page":"Example","title":"Define the simulator","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"The simulator should take a vector of parameters.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"function ricker(r, ϕ, σ; init_n=5, n_iters=200, nburn=50)\n  ϵ = randn(n_iters)*σ\n  nₜ = init_n\n\n  x = Vector{Int}(undef, n_iters)\n  for i in 1:n_iters\n    x[i]  = rand(Poisson(ϕ*nₜ))\n    nₜ = r*nₜ*exp(-nₜ + ϵ[i])\n  end\n  x[nburn+1:n_iters]\nend","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Make parameter input a vector:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"ricker(θ::Vector{Float64}) = ricker(θ...)\nnothing #hide","category":"page"},{"location":"example/#Summary-function","page":"Example","title":"Summary function","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"If no summary statistic function is used, then the summary defualts to the identity. However, a summary function can be specified that summarises the output of the simulator in to a vector.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"function ricker_summary(x)\n    if all(x.==0)\n      return [0., 0., length(x), 0, 0, 0, 0]\n    else\n      s = [mean(x[x.>0]),\n        median(x[x.>0]),\n        sum(x.==0),\n        sum(x.>10),\n        autocov(x, [1, 2, 3])...]\n      return s\n    end\n  end\nnothing #hide","category":"page"},{"location":"example/#Ground-truth","page":"Example","title":"Ground truth","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"As this is a toy example, we will generate \"true\" parameters, alongside a \"pseudo-observed\" simulated dataset.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"θ_true = [6, 1, 0.6]\nx_true = ricker(θ_true)\ns_true = ricker_summary(x_true)\nnothing #hide","category":"page"},{"location":"example/#The-prior","page":"Example","title":"The prior","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Priors can either be multivariate distribution or be specified as a Product distribution (for independent priors for each parameter), in either case using Distributions.jl Below a Product distribution is used.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"prior = Product([LogNormal(2, 0.4), Uniform(0, 5), Uniform(0, 5)])\nnothing #hide","category":"page"},{"location":"example/#LocalPosterior","page":"Example","title":"LocalPosterior","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"The local regression MCMC technique estimates the gradient and Hessian of the likelihood at each iteration. To achieve this rather than carrying out many simulations at a single parameter value to estimate the likelihood (as in standard synthetic likelihood), many simulations from a \"local\" area around the current θ value must be used. One can sample parameters consistent with the data using LocalLikelihood. However here we consider Bayesian inference, so will use LocalPosterior. This interanally uses LocalLikelihood to estimate the the gradient and Hessian of the likelihood as before, and then uses automatic differentiation of the prior to get the gradient and Hessian of the prior. These can then be used to calculcate the gradient and Hessian of the posterior.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"local_posterior = LocalPosterior(;\n  simulator = ricker,\n  summary = ricker_summary,\n  s_true,\n  n_sim = 1000,\n  prior,\n)\nnothing #hide","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"The above object describes how we want to estimate the posterior at each step in the sampler. The other important parameter P is the initial proposal distribution. This is what is used to generate n_sim peturbed parameters to make the local regressions possible (we require variance in the parameters). This parameter isn't shown as it is left to the default (inferred from the prior). The proposal adapts based on the Hessian estimate at each iteration.","category":"page"},{"location":"example/#The-sampler","page":"Example","title":"The sampler","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"We can then sample from the posterior. Below I will use the Riemannian Unadjusted Langevin sampler (RiemannianULA) with a step size of 0.1. A rough explanation of this sampler is that it uses a Newton update, and adds some noise at each iteration.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"rula = RiemannianULA(0.1)\ninit_θ = [8, 4, 0.1]\nn_steps = 2000\ndata = run_sampler!(rula, local_posterior; init_θ, n_steps)","category":"page"},{"location":"example/#Plotting-the-results","page":"Example","title":"Plotting the results","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"StatsPlots.jl provides most the tools required for plotting results.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"θ_names = [\"r\" \"ϕ\" \"σ\"]\nplot(data.θ, layout = 3, xlabel = θ_names, labels = false)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"We can remove the burn in and plot the marginal densities. The package provides plot_prior_posterior_density to achieve this simply.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"samples = data.θ[1001:end, :]\nplot_prior_posterior_density(\n  prior, samples, θ_true; θ_names\n)\n\ncorrplot(samples, labels = θ_names)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"This page was generated using Literate.jl.","category":"page"}]
}
