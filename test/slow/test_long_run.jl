# A long busy simulation (src/main.jl and everything it calls), checked as it goes.
#
# Contracts: the system stays consistent over many grid rebuilds, insertions and promotions;
# agents are never lost; and the membrane's area accounts exactly for what was inserted: once no
# insertion is in progress, the growth since the start equals the total area of the added
# agents divided by the density.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "long run" begin
    p = busy_params(OmpCF=(insertion_prob=0.3,), insertion=(PP_arrival_rate=20.0,))
    state, ps = random_state(p; OmpA=6, OmpCF=4, LptD=3, BamA=4, LPS=120, L=1.5, num_PP=6, rng=Xoshiro(21))
    counts0 = agent_counts(state)
    area0 = area(state)

    t, n_inconsistent, lost = 0.0, 0, false
    n_prev = state.grid_size.num_agents
    for _ in 1:200
        t = run_steps!(state, ps, 100; t0=t)
        n_inconsistent += !isempty(consistency_errors(state, ps))
        lost |= state.grid_size.num_agents < n_prev
        n_prev = state.grid_size.num_agents
    end
    @test n_inconsistent == 0
    @test !lost

    # finish any insertions in progress, with no new ones starting, then balance the area
    finish = params_with(ps; LPS=(arrival_rate=0.0,), insertion=(PP_arrival_rate=0.0, PP_bind_rate=0.0))
    t = step_until!(s -> num_nascent(s) == 0, state, finish; t0=t)
    counts = agent_counts(state)
    added = sum((getfield(counts, Symbol(type)) - getfield(counts0, Symbol(type))) * disk_area(type_radius(ps, type))
                for type in ("OmpA", "OmpCF", "BamA", "LptD", "LPS"))
    @test added > 0
    @test area(state) - area0 ≈ added / ps.system.density rtol=1e-2
    @test consistency_errors(state, finish) == String[]
end
