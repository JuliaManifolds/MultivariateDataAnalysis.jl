
"""
    ICA_425 <: MDAModelType

ICA model, 4.25 from Trendafilov 2021.
"""
struct ICA_425 <: MDAModelType end

function make_objective(model::MDASubspaceModel{ICA_425}, data)
    F, D, At = svd(data)
    return MDAObjective(model, F)
end

function (obj::MDAObjective{<:MDASubspaceModel{ICA_425}})(p)
    #println("obj: ", norm(obj.data * p))
    n, r = representation_size(obj.model.M)
    Jn = Matrix{eltype(p)}(I(n)) - fill(1 / n, n, n)
    FQ = obj.data * p
    FQdFQ = FQ .* FQ
    S = 1 / (n - 1) * FQdFQ' * Jn * FQdFQ
    ErmIr = fill(1.0, r, r) - I(r)
    return tr((S .* S) * ErmIr)
end

function make_gradient(model::MDASubspaceModel{ICA_425}, data)
    F, D, At = svd(data)
    return MDAGradient(model, F)
end

function (obj::MDAGradient{<:MDASubspaceModel{ICA_425}})(X, p)
    # can be derived using matrixcalculus.org
    # tr((((F*Q) .* (F*Q))' * (eye - matrix(1/a)) * ((F*Q) .* (F*Q))) .* (((F*Q) .* (F*Q))' * (eye - matrix(1/a)) * ((F*Q) .* (F*Q))) * (matrix(1)-eye) )

    n, r = representation_size(obj.model.M)
    F = obj.data
    T0 = F * p # FQ
    T1 = T0 .* T0
    T2 = Matrix{eltype(p)}(I(n)) - fill(1 / n, n, n) # Jn
    ErmIr = fill(1.0, r, r) - I(r)

    X .= 8 * F' * ((T2 * T1 * (ErmIr .* (T1' * T2 * T1))) .* T0) / (n - 1)^2
    return X
end
function (obj::MDAGradient{<:MDASubspaceModel{ICA_425}})(::AbstractManifold, p)
    X = similar(p)
    return obj(X, p)
end

function fit(msm::MDASubspaceModel{<:ICA_425}, data)
    # centering
    data_center = data .- mean(data; dims = 1)
    oo = make_objective(msm, data_center)
    og = make_gradient(msm, data_center)

    #println("nc = ", num_components)
    n, r = representation_size(msm.M)

    x0 = 1.0 * I[1:n, 1:r]
    # Manopt.check_gradient(msm.M, oo, og, x0, rand(msm.M; vector_at=x0); plot=true)
    sol = solve_problem_qn(msm.M, oo, og, x0)

    return MDASubspaceModelFit(msm, sol)
end
