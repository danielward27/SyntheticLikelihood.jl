# MCMC to explore the likelihood

# TODO Define local area function? Or always normal?



mutable struct MCMC
    θ::AbstractVector
    objective::Function
    propose::Function
    counter::Int
end





# One step?
function step(MCMC)
    error("Not implemented")
end
