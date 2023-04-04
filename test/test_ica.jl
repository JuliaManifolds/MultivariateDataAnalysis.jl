
using Test
using LinearAlgebra
using Manifolds
using MultivariateDataAnalysis
using StableRNGs
using Statistics

rng = StableRNG(15678)

function generate_test_data_ica(rng, n, k, m)
    t = range(0.0, step = 10.0, length = n)
    s1 = sin.(t * 2)
    s2 = s2 = 1.0 .- 2.0 * Bool[isodd(floor(Int, x / 3)) for x in t]
    s3 = Float64[mod(x, 5.0) for x in t]

    s1 += 0.1 * randn(rng, n)
    s2 += 0.1 * randn(rng, n)
    s3 += 0.1 * randn(rng, n)

    S = hcat(s1, s2, s3)'
    @assert size(S) == (k, n)
    A = randn(rng, m, k)

    X = A * S
    mv = vec(mean(X, dims = 2))
    @assert size(X) == (m, n)
    C = cov(X, dims = 2)
    return X, mv, C, A
end

@testset "ICA 425" begin
    # sources
    n = 1000
    k = 3
    m = 8
    X, μ, C, A = generate_test_data_ica(rng, n, k, m)
    M = Grassmann(m, k)
    model = MDASubspaceModel(MultivariateDataAnalysis.ICA_425(), M)
    mf = fit(model, X)
    x_ref = [
        -0.9855017955156906 -0.010518834296820762 0.0033382811566106604
        0.000988090063745936 -0.9634005664828102 0.04115835214911599
        -0.02264412154254251 0.026701842076455366 -0.9625560794574388
        0.07077106872415313 0.13380881628651875 0.13509167374360886
        0.07061284153987082 -0.21430775978074398 -0.19785868547046154
        -0.11033945705511704 0.020442895689254524 0.11492196180045273
        0.022707961352070868 0.07474106613404463 0.022123875192779037
        -0.0747483352474409 0.03462600303390352 0.02619726967175461
    ]
    @test distance(M, x_ref, mf.p) < 0.45

end
