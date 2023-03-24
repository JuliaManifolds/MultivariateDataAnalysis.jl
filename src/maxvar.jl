


"""
    MaxVar <: MDAModelType

Select variance-maximizing subspace.
"""
struct MaxVar <: MDAModelType end


struct Orthomax <: MDAModelType
    λ::Float64
end

Varimax() = Orthomax(1.0)
Quartimax() = Orthomax(0.0)


make_objective(model::MDASubspaceModel{MaxVar}, data) = MDAObjective(model, data)

function make_objective(model::MDASubspaceModel{Orthomax}, data)
    F, D, At = svd(data)
    A = transpose(At)
    return MDAObjective(model, (; λAtA = model.model_type.λ * At * A))
end

function (obj::MDAObjective{<:MDASubspaceModel{MaxVar}})(p)
    k = obj.data * p
    #println(norm(obj.data - k * p')^2)
    return norm(obj.data - k * p')^2
end


function (obj::MDAObjective{<:MDASubspaceModel{<:Orthomax}})(p)
    pdim = size(p, 2)
    diagsum = zeros(pdim)
    for a_i in eachcol(p)
        Wi = obj.data.λAtA .- pdim .* a_i * a_i'
        diagsum .+= diag(p' * Wi * p) .^ 2
    end
    #println(ret)
    return -sum(diagsum)
end

make_gradient(model::MDASubspaceModel{MaxVar}, data) = MDAGradient(model, data)
function make_gradient(model::MDASubspaceModel{Orthomax}, data)
    return MDAGradient(model, (; λAtA = model.model_type.λ * data' * data))
end


# inplace variant
function (obj::MDAGradient{<:MDASubspaceModel{MaxVar}})(X, p)
    # can be derived using matrixcalculus.org
    XA = obj.data * p
    T0 = obj.data - XA * p'
    XtT0 = obj.data' * T0
    X .= -2 .* XtT0 * p .- 2 .* T0' * XA

    project!(obj.model.M, X, p, X)
    #println(norm(X))
    return X
end


# inplace variant
function (obj::MDAGradient{<:MDASubspaceModel{<:Orthomax}})(X, p)
    # can be derived using matrixcalculus.org
    pdim = size(p, 2)
    X .= 0
    for a_i in eachcol(p)
        Wi = obj.data.λAtA .- pdim .* a_i * a_i'
        Wip = Wi * p
        X .+= Wip * diag(p' * Wip)
    end
    X .*= -4
    project!(obj.model.M, X, p, X)
    #println(norm(X))
    return X
end

function fit(msm::MDASubspaceModel{MaxVar}, data)

    # centering
    data_center = data .- mean(data; dims = 1)
    num_components = get_num_components(msm.M)
    oo = make_objective(msm, data_center)
    og = make_gradient(msm, data_center)

    #println("nc = ", num_components)

    x0 = eigval_directions(data_center, num_components)
    # Manopt.check_gradient(M, f, g, x0, rand(M; vector_at=x0); plot=true)
    sol = solve_problem_qn(msm.M, oo, og, x0)

    return MDASubspaceModelFit(msm, sol)
end


function fit(msm::MDASubspaceModel{<:Orthomax}, data)

    # centering
    data_center = data .- mean(data; dims = 1)
    num_components = get_num_components(msm.M)
    oo = make_objective(msm, data_center)
    og = make_gradient(msm, data_center)

    #println("nc = ", num_components)

    x0 = eigval_directions(data_center, num_components)
    # Manopt.check_gradient(M, f, g, x0, rand(M; vector_at=x0); plot=true)
    sol = solve_problem_qn(msm.M, oo, og, x0)

    return MDASubspaceModelFit(msm, sol)
end
