
using Test
using LinearAlgebra
using MultivariateDataAnalysis

@testset "MultivariateDataAnalysis.jl" begin
    include("test_ica.jl")
    include("test_maxvar.jl")
end
