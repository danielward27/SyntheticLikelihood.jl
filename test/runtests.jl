using SyntheticLikelihood, Test, SafeTestsets

@time begin
    @time @safetestset "Simulation interface" begin include("simulation_interface_test.jl") end
    @time @safetestset "Likelihood" begin include("likelihood_test.jl") end
    @time @safetestset "MCMC" begin include("mcmc_test.jl") end
    @time @safetestset "Local regression" begin include("local_regression_test.jl") end
    @time @safetestset "Utils" begin include("utils_test.jl") end
end
