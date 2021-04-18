# Documentation

## Sampling
```@autodocs
Private = false
Modules = [SyntheticLikelihood]
Pages = ["samplers.jl"]
```

#### Implementation details
To implement a new sampler:
- Create a subtype of `AbstractSampler`, which takes the hyperparameters.
- Create a subtype of `AbstractSamplerState` which stores the state of the sampler
    at each iteration.
- Implement the `get_init_state` and `update!` functions to update the state
(e.g. parameters, gradients, objective function value).


## Objectives
```@autodocs
Private = false
Modules = [SyntheticLikelihood]
Pages = ["local_approximation_structs.jl"]
```
#### Implementation details
To implement a new objective:
- Create a subtype of `LocalApproximation` defining the hyperpameters.
- Implement `obj_grad_hess` function to estimate the objective, gradient and
    Hessian.

## Matrix regularisation
As the Hessian inferred from local regressions is not guarenteed to be positive
definite, we may need to modify the Hessian (or its inverse) in some way.

```@autodocs
Private = false
Modules = [SyntheticLikelihood]
Pages = ["matrix_regularizers.jl"]
```

## Plotting
Some convenience functions for plotting:
```@autodocs
Private = false
Modules = [SyntheticLikelihood]
Pages = ["plotting.jl"]
```
