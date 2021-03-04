# Simulation interface
This section describes the structure a simulator should have to work with the package. Simulator functions should take a single positional argument which is a vector of parameters (and can have arbitrary key word arguments). Summary functions should take the output from the simulator directly as a single positional argument (and can have arbitrary key word arguments).

The primary function for carrying out simulations is `simulate_n_s`, which simulates `n` sets of summary statistics (or the raw simulator output if no summary is specified).

```@docs
simulate_n_s
```
