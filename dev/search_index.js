var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SyntheticLikelihood","category":"page"},{"location":"#SyntheticLikelihood","page":"Home","title":"SyntheticLikelihood","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SyntheticLikelihood package. The package is currently a work in progress. Simulator functions should take a single positional argument which is a vector of parameters (and can have arbitrary key word arguments). Summary functions should take the output from the simulator directly as a single positional argument (and can have arbitrary key word arguments).","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"man/sampling.md\"]","category":"page"},{"location":"#Function-list","page":"Home","title":"Function list","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Function-descriptions","page":"Home","title":"Function descriptions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [SyntheticLikelihood]","category":"page"},{"location":"#SyntheticLikelihood.AbstractSamplerState","page":"Home","title":"SyntheticLikelihood.AbstractSamplerState","text":"Concrete sampler types should have fields:\n\nθ::AbstractVector\nobjective::Float64\ncounter::Integer\n\n\n\n\n\n","category":"type"},{"location":"#SyntheticLikelihood.BasicState","page":"Home","title":"SyntheticLikelihood.BasicState","text":"Struct for containing the state of sampler at each iteration     for simple samplers (that just require objective evaluation).\n\n\n\n\n\n","category":"type"},{"location":"#SyntheticLikelihood.GradientHessianState","page":"Home","title":"SyntheticLikelihood.GradientHessianState","text":"Struct for containing the state of sampler at each iteration     for samplers which use the gradient and hessian.\n\n\n\n\n\n","category":"type"},{"location":"#SyntheticLikelihood.GradientState","page":"Home","title":"SyntheticLikelihood.GradientState","text":"Struct for containing the state of sampler at each iteration     for simple gradient based samplers.\n\n\n\n\n\n","category":"type"},{"location":"#SyntheticLikelihood.LocalΣ","page":"Home","title":"SyntheticLikelihood.LocalΣ","text":"Struct that contains the estimated local properties of Σ (the covariance matrix of the summary statistics).\n\nFields\n\nΣ The (nₛ×nₛ) (estimated) covariance matrix of the summary statistics.\n∂ The (nₛ×nₛ×n_θ) matrix of estimated first derivitives of Σ.\n\n\n\n\n\n","category":"type"},{"location":"#SyntheticLikelihood.Localμ","page":"Home","title":"SyntheticLikelihood.Localμ","text":"Struct that contains the estimated local properties of μ.\n\nFields\n\nμ::Float64 Mean of the summary statistic.\n∂::Vector{Float64} First derivitive w.r.t. parameters.\n∂²::Matrix{Float64} Second derivitive w.r.t. parameters.\nϵ::Vector{Float64} Residuals.\n\n\n\n\n\n","category":"type"},{"location":"#SyntheticLikelihood.add_state!-Tuple{NamedTuple,AbstractSamplerState,Integer}","page":"Home","title":"SyntheticLikelihood.add_state!","text":"Add data to the data tuple.\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.get_residuals-Tuple{Array{Localμ,1}}","page":"Home","title":"SyntheticLikelihood.get_residuals","text":"Get an array of residuals from a vector of Localμ structs. Returns (length of residuals × number of summary stats) matrix\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.glm_local_Σ-Tuple{}","page":"Home","title":"SyntheticLikelihood.glm_local_Σ","text":"Use a gamma distributed GLM with log link function to estimate the local properties     of Σ. θ should not have a bias term (added internally).\n\nArguments\n\nθ_orig::AbstractVector  Original θ (used for centering).\nθ::AbstractMatrix Peturbed θ from local area.\nϵ::AbstractMatrix Residuals from quadratic regression.\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.init_data_tuple-Tuple{AbstractSamplerState,Array{Symbol,1},Integer}","page":"Home","title":"SyntheticLikelihood.init_data_tuple","text":"Function initialises a named tuple containing Vectors with undefined values. Used with samplers to store results. State just provides an \"example\" state from which to infer types of vectors in the array. Names of the named tuple are the symbols provided.\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.langevin_diffusion-Tuple{AbstractArray{Float64,1}}","page":"Home","title":"SyntheticLikelihood.langevin_diffusion","text":"As above, but the initial state is induced from init_θ, rather than explicitly providing a startings state.\n\nlangevin_diffusion(init_θ; objective, gradient, step_size, n_steps, collect_data)\n\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.langevin_diffusion-Tuple{GradientState}","page":"Home","title":"SyntheticLikelihood.langevin_diffusion","text":"Sample using Langevin diffusion . Uses a discrete time Euler approximation of the Langevin diffusion (unadjusted Langevin algorithm), given by the update update θ := θ - stepsize/2 .* ∇θ .+ ξ. ξ is given by `MvNormal(stepsize)`. Uses a fixed step size.\n\nNote if the aim is to sample from a density function, the negative of its gradient and density should be provided (sampler \"aims\" to minimize the function).\n\nReturns a tuple, (data, state), where data is a NamedTuple of stored data, and state is the state at the final iteration, which can be used to reinitialise the algorithm if desired.\n\nArguments: state::GradientState Initial starting state for sampler. objective::Function Objective function (assumed \"aim\" would be to minimize) gradient::Function Gradient of the objective function with respect to the parameters. step_size::Vector{Float64} Multiplied elementwise by gradient. n_steps::Integer Number of iterations to carry out. collect_data::Vector{Symbol}=[:θ, :objective] Vector of symbols, denoting the     items in the state to store at each iteration.\n\nlangevin_diffusion(state; objective, gradient, step_size, n_steps, collect_data)\n\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.linear_regression-Tuple{AbstractArray{T,2} where T,AbstractArray{T,1} where T}","page":"Home","title":"SyntheticLikelihood.linear_regression","text":"Carry out linear regression. X should have a bias column. Returns tuple (β, ŷ).\n\nlinear_regression(X, y)\n\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.pairwise_combinations-Tuple{Integer}","page":"Home","title":"SyntheticLikelihood.pairwise_combinations","text":"Pairwise combinations (for quadratic regression). n=5 would return all the pairwise combinations between 1:5 (including matched terms e.g. [1,1]).\n\npairwise_combinations(n)\n\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.peturb","page":"Home","title":"SyntheticLikelihood.peturb","text":"Peturb a vector using a user specified distribution (often MVN zero mean). Returns array of size (n, length(θ))\n\npeturb(θ, d)\npeturb(θ, d, n)\n\n\nArguments\n\nθ::AbstractVector Parameter to peturb.\nd::Sampleable Distribution from which to sample (see Distributions.jl).\nn::Integer = 1 Number of peturbed vectors to return.\n\n\n\n\n\n","category":"function"},{"location":"#SyntheticLikelihood.quadratic_design_matrix-Tuple{AbstractArray{T,2} where T}","page":"Home","title":"SyntheticLikelihood.quadratic_design_matrix","text":"Design matrix for quadratic regression. Bias term appended as first column internally. Returns a tuple, with the matrix and the corresponding indices multiplied, that give rise to each column. Note, indices [1, 1] corresponds to the bias term (so indices compared to original matrix is shifted).\n\nquadratic_design_matrix(X)\n\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.quadratic_local_μ-Tuple{}","page":"Home","title":"SyntheticLikelihood.quadratic_local_μ","text":"Finds the local behaviour of the summary statistic mean μ. Uses quadratic linear regression to approximate the mean, gradient and hessian around θ_orig. Returns a vector of Localμ structs (see above), with length equal to the number of summary statistics.\n\nquadratic_local_μ(; θ_orig, θ, s)\n\n\nArguments\n\nθ_orig::AbstractVector Original θ.\nθ::AbstractMatrix Peturbed θ (sampled from local area).\ns::AbstractMatrix Corresponding summary statistics to θ.\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.simplify_data-Tuple{NamedTuple}","page":"Home","title":"SyntheticLikelihood.simplify_data","text":"Loop through named tuple and call stack_arrays on any vector whose elements are an array. Used at end of samplers.\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.simulate_n_s-Tuple{AbstractArray{T,1} where T}","page":"Home","title":"SyntheticLikelihood.simulate_n_s","text":"Simulates summary statistics from the model under a fixed parameter vector. n_sim is specified as the number of simulations. Simulations can be run on multiple threads using parallel = true. By defualt no summary statistic function is used (by passing the identity function).\n\nsimulate_n_s(θ; simulator, summary, n_sim, simulator_kwargs, summary_kwargs, parallel)\n\n\nArguments\n\nθ::AbstractVector Parameter vector passed to simulator.\nsimulator::Function Simulator.\nsummary::Function Summary function that takes output of simulator (defualt identity).\nn_sim::Integer Number of simulations.\nsimulator_kwargs Kwargs passed to simulator.\nsummary_kwargs Kwargs passed to summary.\nparallel::Bool = true Whether to run on multiple threads.\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.simulate_n_s-Tuple{AbstractArray{T,2} where T}","page":"Home","title":"SyntheticLikelihood.simulate_n_s","text":"As for above, but a Matrix of parameter values are used, carrying out one     simulation from each row of θ (and hence n_sim is not required).\n\nsimulate_n_s(θ; simulator, summary, simulator_kwargs, summary_kwargs, parallel)\n\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.stack_arrays-Tuple{Array{T,1} where T}","page":"Home","title":"SyntheticLikelihood.stack_arrays","text":"Stacks a vector of consitently sized arrays to make a new array with dimensions (length(x), dim(x[1])...).\n\n\n\n\n\n","category":"method"},{"location":"#SyntheticLikelihood.synthetic_likelihood-Tuple{AbstractArray{T,1} where T}","page":"Home","title":"SyntheticLikelihood.synthetic_likelihood","text":"Evaluates synthetic likelhood of observed data for a fixed parameter vector     using a multivariate Gaussian assumption as in (Simon Wood, 2010).\n\nsynthetic_likelihood(θ; simulator, summary, s_true, n_sim, simulator_kwargs, summary_kwargs)\n\n\nArguments\n\nθ::AbstractVector Parameter vector passed to simulator.\nsimulator::Function Simulator.\nsummary::Function Summary function that takes output of simulator (defualt identity).\ns_true::AbstractVector Observed summary statistics.\nn_sim::Integer Number of simulations to use.\nsimulator_kwargs Kwargs splatted in simulator.\nsummary_kwargs Kwargs splatted in summary.\n\n\n\n\n\n","category":"method"},{"location":"man/sampling/#Sampling","page":"Sampling","title":"Sampling","text":"","category":"section"},{"location":"man/sampling/","page":"Sampling","title":"Sampling","text":"Samplers can either be initialised using an initial parameter vector θ, or using an AbstractSamplerState object. The latter is returned at the end of a sampling run, and provides a convenient way to continue sampling further.","category":"page"},{"location":"man/sampling/","page":"Sampling","title":"Sampling","text":"Below is an example to sample from a multivariate normal density using the discretized langevin diffusion (Unadjusted Langevin Algorithm).","category":"page"},{"location":"man/sampling/","page":"Sampling","title":"Sampling","text":"using SyntheticLikelihood, Distributions, Gadfly\n\nd = MvNormal([10 5; 5 10])\ninit_θ = [-15., -15]\nobjective(θ) = -logpdf(d, θ)\ngradient(θ) = -gradlogpdf(d, θ)\n\ndata, state = langevin_diffusion(\n        init_θ; objective, gradient, step_size = [1.,1],\n        n_steps = 1000, collect_data = [:θ])\n\nθ_samples = data[:θ]\nx = y = range(-20, 20; length=50)\n\ncontours = layer(z = (x, y) -> logpdf(d, [x,y]), x = x, y = y, Geom.contour)\npoints = layer(x = θ_samples[:,1], y = θ_samples[:,2], Geom.point,\n                Theme(alphas=[0.2], default_color = \"black\"))\n\nplot(contours, points)","category":"page"}]
}
