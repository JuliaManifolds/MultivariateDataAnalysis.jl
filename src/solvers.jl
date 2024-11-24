

function solve_problem_qn(M::AbstractManifold, oo, og, x0)
    stopping_criterion =
        StopAfterIteration(100) |
        StopWhenGradientNormLess(1e-6) |
        StopWhenChangeLess(M, 1e-10)

    mgo = ManifoldGradientObjective(oo, og; evaluation = InplaceEvaluation())
    cmgo = ManifoldCachedObjective(M, mgo; p = x0)
    mp = DefaultManoptProblem(M, cmgo)

    qns = QuasiNewtonState(M; p = x0, stopping_criterion)
    solret = solve!(mp, qns)
    #println(solret)
    sol = get_solver_return(solret)
    return sol
end

function solve_problem_cpp(M::AbstractManifold, oo, op, x0)
    stopping_criterion =
        StopAfterIteration(100) |
        StopWhenGradientNormLess(1e-6) |
        StopWhenChangeLess(M, 1e-10)

    mpo = ManifoldProximalMapObjective(oo, op; evaluation = InplaceEvaluation())
    dmpo = ManifoldCachedObjective(M, mpo; p = x0)
    dmp = DefaultManoptProblem(M, dmpo)
    cpps = CyclicProximalPointState(
        M,
        p;
        stopping_criterion = stopping_criterion,
        λ = λ,
        evaluation_order = evaluation_order,
    )
    solret = solve!(dmp, cpps)
    sol = get_solver_return(solret)
    return sol
end

function solve_problem_subgradient(M::AbstractManifold, oo, og, x0; max_iter::Int = 2000)
    stopping_criterion = StopAfterIteration(max_iter) | StopWhenChangeLess(M, 1e-10)
    sgo = ManifoldSubgradientObjective(oo, og; evaluation = InplaceEvaluation())
    dmp = DefaultManoptProblem(M, sgo)
    stepsize = Manopt.DecreasingStepsize(M; factor = 0.99)

    sgs = SubGradientMethodState(
        M;
        p = x0,
        stopping_criterion = stopping_criterion,
        stepsize = stepsize,
    )

    solret = solve!(dmp, sgs)
    # println(solret)
    sol = get_solver_return(solret)
    return sol
end
