module MultivariateDataAnalysis

using Markdown: @doc_str
using LinearAlgebra

using ManifoldsBase
using ManifoldsBase: get_parameter
using Manifolds

using Manopt
using Manopt: get_solver_return
using Optim

using LRUCache

using Requires

using StatsAPI
import StatsAPI: fit, predict, coef, dof, r2, reconstruct

using Statistics

using RecursiveArrayTools

abstract type MDAModelType end


"""
    MDASubspaceModel{TMT<:MDAModelType,TM<:AbstractManifold}

`model_type` descibes the objective for subspace selection, while `M`
states from what manifold the solution to the problem comes.
"""
struct MDASubspaceModel{TMT<:MDAModelType,TM<:AbstractManifold}
    model_type::TMT
    M::TM
end

struct MDAObjective{TM<:MDASubspaceModel,TD}
    model::TM
    data::TD
end

function (obj::MDAObjective)(::AbstractManifold, p)
    return obj(p)
end

struct MDAGradient{TM<:MDASubspaceModel,TD}
    model::TM
    data::TD
end

function (obj::MDAGradient{<:MDASubspaceModel})(::AbstractManifold, X, p)
    return obj(X, p)
end


struct MDASubgradient{TM<:MDASubspaceModel,TD}
    model::TM
    data::TD
end

function (obj::MDASubgradient)(::AbstractManifold, X, p)
    # println(hash(p))
    return obj(X, p)
end

function get_num_components(M::Grassmann)
    return get_parameter(M.size)[2]
end
function get_num_components(M::Manifolds.Stiefel)
    return get_parameter(M.size)[2]
end
function get_num_components(M::Oblique)
    return get_parameter(M.size)[2]
end

function eigval_directions(data, num_components::Int)
    C = cov(data' * data)
    irange = (size(C, 1) - num_components + 1):size(C, 1)
    eg = eigen(Symmetric(C), irange)
    evectors_in_decreasing_eigval_order = reverse(eg.vectors; dims = 2)
    return evectors_in_decreasing_eigval_order
end


"""
    MDASubspaceModelFit{TM<:MDASubspaceModel,TP}

Fitted model, with parameters described by `p`.
"""
struct MDASubspaceModelFit{TM<:MDASubspaceModel,TP} <: StatsAPI.StatisticalModel
    model::TM
    p::TP
end

# generic utilities

include("solvers.jl")

# specific methods

include("clf.jl")
include("ica.jl")
include("maxvar.jl")
#include("SPCA.jl")
#include("SPCA copy.jl")
#include("SPCA copy 1.jl")


export MDASubspaceModel, MDASubspaceModelFit, MaxVar

export fit, predict, reconstruct

end # module
