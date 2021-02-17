using SyntheticLikelihood, Test, SafeTestsets

@time begin
    @time @safetestset "Simulation interface" begin include("simulation_interface_test.jl") end
end
