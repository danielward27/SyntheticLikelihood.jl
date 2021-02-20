using SyntheticLikelihood, Test

X = [1 1 2; 1 4 3]
expected_result = [1 1 1 2 2 4; 1 4 16 3 12 9]
@test quadratic_transform(X)[1] == expected_result
