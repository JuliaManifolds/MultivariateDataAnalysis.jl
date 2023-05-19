# MultivariateDataAnalysis.jl

Multivariate data analysis using geometric algorithms made easy!

The package __MultivariateDataAnalysis__ aims to provide an easy to use interface for a wide variety of multivariate statistical models like Principal Component Analysis, Linear Discriminant Analysis, Independent Component Analysis, VARIMAX and their variants, especially those formulated using geometric algorithms. It extends the [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl/) interface.

It is similar in scope to [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl), although it __MultivariateDataAnalysis__ aims to provide a wider variety of methods that require additional dependencies on optimization libraries.

Example usage:

```julia
using Manifolds, MultivariateDataAnalysis, RDatasets

data = Array(dataset("datasets", "iris")[!, Not(:Species)])
model = MDASubspaceModel(MaxVar(), Grassmann(size(data, 2), 2))
mf = fit(model, data)
predict(mf, [5.0, 3.0, 2.0, 1.0])
```
