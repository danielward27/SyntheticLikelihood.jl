using SyntheticLikelihood, Test

X = [1 2; 4 3]
expected_result = [1 1 1 2 2 4; 1 4 16 3 12 9]
@test quadratic_transform(X)[1] == expected_result

# Quadratic regression should be able to represent true quadratic perfectly
function quadratic(x1::Vector{Float64}, x2::Vector{Float64})
    x1 .+ 2x2 .+ 3x1.*x2 .+ 4x1.^2 .+ 5x2.^2
end

X = rand(10, 2)
y = quadratic(X[:, 1], X[:, 2])

X, combinations = quadratic_transform(X)
β, ŷ = linear_regression(X, y)
@test isapprox(y, ŷ)
