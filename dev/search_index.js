var documenterSearchIndex = {"docs":
[{"location":"local_regression/#Local-regression","page":"Local regression","title":"Local regression","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"Local regressions can be used to estimate the gradient and hessian of the likelihood, which can be used to improve the sampling efficieny of synthetic likelihood.","category":"page"},{"location":"local_regression/#Example","page":"Local regression","title":"Example","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"Here we consider a simple example, in which we infer the mean of a 10-dimensional multivariate normal distribution, using simulations from the distribution.","category":"page"},{"location":"local_regression/#Define-the-simulator","page":"Local regression","title":"Define the simulator","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"The simulator must take a single positional argument, which is the parameter vector:","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"using SyntheticLikelihood, Distributions, Plots\n\n# Define the simulator\nfunction simulator(θ::Vector{Float64})\n  @assert length(θ) == 10\n  d = MvNormal(θ, sqrt(0.1))\n  rand(d)\nend\n\nnothing # hide","category":"page"},{"location":"local_regression/#Ground-truth","page":"Local regression","title":"Ground truth","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"The \"true\" parameters which we will aim to estimate, is just a vector of zeros. We can use this to generate a pseudo-observed data set s_true.","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"θ_true = zeros(10)\ns_true = simulator(θ_true)","category":"page"},{"location":"local_regression/#Defining-how-to-estimate-the-likelihood","page":"Local regression","title":"Defining how to estimate the likelihood","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"We can then define the hyperparameters for estimating the likelihood using local regression using LocalLikelihood.","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"local_likelihood = LocalLikelihood(;\n  simulator, s_true,\n  P = MvNormal(fill(0.5, 10)),\n  n_sim = 1000\n)\nnothing # hide","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"Note that if required a summary function can optionally be specified here, to summarise the output of the simulator.","category":"page"},{"location":"local_regression/#Defining-sampling-method","page":"Local regression","title":"Defining sampling method","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"We can then define how to sample from the distribution. Below I will use the RiemannianULA sampler with a step size of 0.1.","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"pULA = RiemannianULA(0.1)\nnothing # hide","category":"page"},{"location":"local_regression/#Sampling","page":"Local regression","title":"Sampling","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"We can now define some initial parameter values, init_θ, and sample from the distribution using run_sampler!:","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"init_θ = convert(Vector{Float64}, 1:10)\ndata = run_sampler!(pULA, local_likelihood; init_θ, n_steps = 500)\nnothing # hide","category":"page"},{"location":"local_regression/#Plot-the-samples","page":"Local regression","title":"Plot the samples","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"param_names = reshape([\"θ$i\" for i in 1:10], (1,10))\nplot(data.θ, label = param_names)","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"We can see that after the burn in period, samples are generally centered around the true parameter values (all zeros). More specifically, they are centered around s_true in this case, which are generally around zero.","category":"page"},{"location":"local_regression/#Bayesian-inference","page":"Local regression","title":"Bayesian inference","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"Given a prior, it is also simple to sample from the posterior instead of the likelihood. The prior should be specified using the distributions from Distributions.jl. A multivariate distribution can be used, or alternatively the prior can be formed from independent univariate priors using a Product distribution from Distributions.jl. For this example the prior is a multivariate normal centered around 5, with no correlation structure, and σ=0.5:","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"prior = MvNormal(fill(5, 10), 0.5)\nnothing # hide","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"We can then define our objective using LocalPosterior and run the sampler again:","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"local_posterior = LocalPosterior(prior, local_likelihood)\ndata = run_sampler!(pULA, local_posterior; init_θ, n_steps = 500)\nplot(data.θ, label = param_names)","category":"page"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"Note internally, this uses LocalLikelihood to estimate the the gradient and Hessian of the likelihood as before, and then uses automatic differentiation of the prior to get the gradient and Hessian of the prior. These can then be used to calculcate the gradient and Hessian of the posterior.","category":"page"},{"location":"local_regression/#Currently-available-\"objectives\"","page":"Local regression","title":"Currently available \"objectives\"","text":"","category":"section"},{"location":"local_regression/","page":"Local regression","title":"Local regression","text":"LocalLikelihood\nLocalPosterior","category":"page"},{"location":"local_regression/#SyntheticLikelihood.LocalLikelihood","page":"Local regression","title":"SyntheticLikelihood.LocalLikelihood","text":"Contains the hyperparameters for getting a local approximation of the negative log-likelihood surface using local regressions.\n\nsimulator\nsummary\ns_true\nP\nInitial distribution used to peturb the parameter value.\nn_sim\nThe number of peturbed points to use for the local regression.\nP_regularizer\nAdaptive proposal distribution. Should not be set manually.\n\n\n\n\n\n","category":"type"},{"location":"local_regression/#SyntheticLikelihood.LocalPosterior","page":"Local regression","title":"SyntheticLikelihood.LocalPosterior","text":"Contains the hyperparameters for getting a local approximation of the posterior.\n\nprior\nsimulator\nsummary\ns_true\nP\nn_sim\nP_regularizer\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SyntheticLikelihood","category":"page"},{"location":"#SyntheticLikelihood","page":"Home","title":"SyntheticLikelihood","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SyntheticLikelihood package. The package is currently a work in progress.","category":"page"},{"location":"samplers/#Samplers","page":"Samplers","title":"Samplers","text":"","category":"section"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"To sample from a distribution, first a sampler object should be created. The currently available samplers are shown below:","category":"page"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"ULA\nRiemannianULA","category":"page"},{"location":"samplers/#SyntheticLikelihood.ULA","page":"Samplers","title":"SyntheticLikelihood.ULA","text":"Sampler for unadjusted langevin algorithm. Uses a discrete time Euler approximation of the langevin diffusion, given by the update θ := θ - η/2 * ∇ + ξ, where ξ is given by N(0, ηI).\n\nstep_size\n\n\n\n\n\n","category":"type"},{"location":"samplers/#SyntheticLikelihood.RiemannianULA","page":"Samplers","title":"SyntheticLikelihood.RiemannianULA","text":"Sampler object for Riemannian ULA. Uses the update: θ := θ - ϵ²H⁻¹*∇ - ϵ√H⁻¹ z, where z ∼ N(0, I).\n\nstep_size\n\n\n\n\n\n","category":"type"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"The sampler object defines the hyperparameters of the sampler, and a function obj_grad_hess, which takes θ and returns a ObjGradHess object, with fields objective, gradient and hessian. The gradient and hessian defualt to nothing if not required. This approach seems a bit convoluted (compared to e.g. seperately passing objective, gradient and Hessian functions), but it facilitates reusing calculations shared between calculating the objective, gradient and hessian, if desired. The aim should be to explore around the minima of the function, so the objective could be the negative log-posterior, for example.","category":"page"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"The sampler object can then passed to run_sampler!, to sample from the distribution:","category":"page"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"run_sampler!","category":"page"},{"location":"samplers/#SyntheticLikelihood.run_sampler!","page":"Samplers","title":"SyntheticLikelihood.run_sampler!","text":"Run the sampling algorithm. Data to collect at each iteration is specified by collect_data, and should be a subset of [:θ, :objective, :gradient, :hessian, :counter].\n\nrun_sampler!(sampler, local_approximation; init_θ, n_steps, collect_data)\n\n\nReturns a tuple, with keys matching collect_data.\n\n\n\n\n\n","category":"function"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"Below is an example to sample from a multivariate normal density using the discretized ULA diffusion (Unadjusted ULA Algorithm). In the below example a summary function is not used (it is left to defualt to the identity), so inference is performed on the raw simulator output.","category":"page"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"using SyntheticLikelihood, Distributions, Plots\n\n# Sample from MVN\nd = MvNormal([10 5; 5 10])\nfunction obj_grad_hess(θ)\n    ObjGradHess(objective = -logpdf(d, θ),\n                       gradient = -gradlogpdf(d, θ))\nend\n\ninit_θ = [-15., -15]\nn_steps = 1000\n\nULA = ULA(1., obj_grad_hess)\ndata = run_sampler!(ULA, init_θ, n_steps, [:θ, :counter])\n\nθ_samples = data[:θ]\n\n# Plot samples\nx = y = range(-20, 20; length=50)\nf(x, y) = -logpdf(d, [x,y])\nX = repeat(reshape(x, 1, :), length(y), 1)\nY = repeat(y, 1, length(x))\nZ = map(f, X, Y)\np = contour(x, y, f)\nscatter!(θ_samples[:,1], θ_samples[:,2], legend = false)","category":"page"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"Note that here a determinisic obj_grad_hess function is used. Generally, in simulation-based-inference this would not be available, and hence this can be replaced for example with local_likelihood, which uses local regressions to approximate the gradient and Hessian of the likelihood function.","category":"page"},{"location":"samplers/#Implementation-details","page":"Samplers","title":"Implementation details","text":"","category":"section"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"To implement a new sampler, each sampler must:","category":"page"},{"location":"samplers/","page":"Samplers","title":"Samplers","text":"Be a subtype of AbstractSampler.\nHave fields containing the hyperparameters.\nHave an update! method, taking the sampler the LocalApproximation and the SamplerState object as arguments e.g. update!(sampler::MySampler, local_approximation::LocalApproximation, state::SamplerState). This updates the state (parameters, gradients objective function value etc, and sampler object if applicable).","category":"page"}]
}