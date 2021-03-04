using SyntheticLikelihood
using Documenter

DocMeta.setdocmeta!(SyntheticLikelihood, :DocTestSetup, :(using SyntheticLikelihood); recursive=true)

makedocs(;
    modules=[SyntheticLikelihood],
    authors="Daniel Ward",
    repo="https://github.com/danielward27/SyntheticLikelihood.jl/blob/{commit}{path}#{line}",
    sitename="SyntheticLikelihood.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://danielward27.github.io/SyntheticLikelihood.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Simulation interface" => "simulation_interface.md",
        "Local regression" => "local_regression.md",
        "Samplers" => "samplers.md"
    ],
)

deploydocs(;
    repo="github.com/danielward27/SyntheticLikelihood.jl",
    devbranch = "main",
)
