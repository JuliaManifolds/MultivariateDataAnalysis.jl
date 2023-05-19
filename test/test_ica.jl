
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
    X_center = X .- mean(X; dims = 1)
    oo = MultivariateDataAnalysis.make_objective(model, X_center)

    @test oo(mf.p) < 1e-10

end
