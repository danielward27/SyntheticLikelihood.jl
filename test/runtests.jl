using SyntheticLikelihood, Test, SafeTestsets

@time begin
    @time @safetestset "Simulate n s" begin include("simulate_n_s_test.jl") end
    @time @safetestset "Matrix regularizers" begin include("matrix_regularizers_test.jl") end
    @time @safetestset "Likelihood" begin include("likelihood_test.jl") end
    @time @safetestset "Samplers" begin include("samplers_test.jl") end
    @time @safetestset "Local regression" begin include("local_regression_test.jl") end
    @time @safetestset "Utils" begin include("utils_test.jl") end
end
