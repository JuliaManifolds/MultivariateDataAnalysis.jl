using Manifolds, ManifoldsBase, MultivariateDataAnalysis, Documenter

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(prettyurls = false, assets = ["assets/favicon.ico"]),
    modules = [MultivariateDataAnalysis],
    authors = "Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename = "MultivariateDataAnalysis.jl",
    pages = ["Home" => "index.md", "Library" => "library.md", "Models" => "models.md"],
)
deploydocs(
    repo = "github.com/JuliaManifolds/MultivariateDataAnalysis.jl.git",
    push_preview = true,
    devbranch = "main",
)
