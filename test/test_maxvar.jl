
using Test
using LinearAlgebra
using Manifolds
using MultivariateDataAnalysis
using StableRNGs
using Statistics

rng = StableRNG(15678)

function generate_test_data_maxvar(rng, n, k, m)
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

    X = (A * S)'
    mv = vec(mean(X, dims = 2))
    @assert size(X) == (n, m)
    C = cov(X, dims = 2)
    return X, mv, C, A
end

@testset "maximum variance" begin
    n = 1000
    k = 3
    m = 8
    X, μ, C, A = generate_test_data_maxvar(rng, n, k, m)
    M = Grassmann(m, k - 1)
    model = MDASubspaceModel(MultivariateDataAnalysis.MaxVar(), M)
    mf = fit(model, X)
    data_center = X .- mean(X; dims = 1)
    obj = MultivariateDataAnalysis.make_objective(model, data_center)
    @test obj(mf.p) < 160.4
    @test predict(mf, X[1, :]) == mf.p' * X[1, :]
    @test reconstruct(mf, [1, 2]) == mf.p * [1, 2]
end
