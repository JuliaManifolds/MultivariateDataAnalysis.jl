

"""
    CLF_A <: MDAModelType

Component Loss Function, 4.22 from Trendafilov 2021.
"""
struct CLF_A <: MDAModelType end

"""
    CLF_B <: MDAModelType

Component Loss Function, 4.23 from Trendafilov 2021.
"""
struct CLF_B <: MDAModelType end

function make_objective(model::MDASubspaceModel{<:Union{CLF_A,CLF_B}}, data)
    return MDAObjective(model, data)
end

function (obj::MDAObjective{<:MDASubspaceModel{CLF_A}})(p)
    #println("obj: ", norm(obj.data * p))
    return norm(obj.data * p)
end

function (obj::MDAObjective{<:MDASubspaceModel{CLF_B}})(p)
    #println("obj: ", norm(obj.data * p))
    return norm(obj.data / p')
end

function (obj::MDASubgradient{<:MDASubspaceModel{CLF_A}})(X, p)
    # can be derived using matrixcalculus.org
    X .= obj.data' * sign.(obj.data * p)

    project!(obj.model.M, X, p, X)
    #println(norm(X))
    return X
end

function (obj::MDASubgradient{<:MDASubspaceModel{CLF_B}})(X, p)
    # can be derived using matrixcalculus.org
    X .= obj.data' * sign.(obj.data * p)

    project!(obj.model.M, X, p, X)
    #println(norm(X))
    return X
end


function make_subgradient(model::MDASubspaceModel{<:Union{CLF_A,CLF_B}}, data)
    return MDASubgradient(model, data)
end


function fit(msm::MDASubspaceModel{<:Union{CLF_A,CLF_B}}, data)

    # centering
    data_center = data .- mean(data; dims = 1)
    num_components = get_num_components(msm.M)
    oo = make_objective(msm, data_center)
    osg = make_subgradient(msm, data_center)

    #println("nc = ", num_components)

    x0 = eigval_directions(data_center, num_components)
    # Manopt.check_gradient(M, f, g, x0, rand(M; vector_at=x0); plot=true)
    sol = solve_problem_subgradient(msm.M, oo, osg, x0)

    return MDASubspaceModelFit(msm, sol)
end
