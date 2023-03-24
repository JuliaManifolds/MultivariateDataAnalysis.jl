
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
    n, r = size(obj.data)
    Jn = Matrix{eltype(p)}(I(n)) - fill(1 / n, n, n)
    FQ = obj.data * p
    FQdFQ = FQ .* FQ
    S = 1 / (n - 1) * FQdFQ' * Jn * FQdFQ
    ErmIr = fill(1.0, r, r) - I(r)
    return tr((S .* S) * ErmIr)
end

function (obj::MDAGradient{<:MDASubspaceModel{<:ICA_425}})(X, p)
    # can be derived using matrixcalculus.org
    # tr(((F*Q) .* (F*Q))' * (eye - matrix(1/a)) * ((F*Q) .* (F*Q)) .* ((F*Q) .* (F*Q))' * (eye - matrix(1/a)) * ((F*Q) .* (F*Q)) * (matrix(1)-eye) )
end
