# MCMC to explore the likelihood

# TODO Define local area function? Or always normal?



@with_kw mutable struct MCMC
    θ::Vector
    objective::Function
    propose::Function
    counter::Int
end





# One step?
function step(MCMC)
    error("Not implemented")
end
