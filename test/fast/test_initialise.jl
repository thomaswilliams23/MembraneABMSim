# System initialisation (src/initialise.jl, src/sim_state.jl): initialise_sim with the
# "specify_positions" and "random" methods, equilibration and shrinkage. (Restarting from a
# checkpoint is covered in test_integration.jl.)
#
# Contracts: specified positions are used exactly, missing ones are filled at random inside the
# domain, and invalid position lists are rejected; every method gives the requested agents in a
# consistent state, assembled or not as asked; equilibration relieves overlaps; shrinkage
# shrinks the domain round by round while holes remain, up to the round limit.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

pos(v) = [SVector{2, Float64}(q...) for q in v]
positions(; OmpA=nothing, OmpCF=nothing, LptD=nothing, BamA=nothing, LPS=nothing) =
    M.InitPositions((isnothing(v) ? nothing : pos(v) for v in (OmpA, OmpCF, LptD, BamA, LPS))...)

# initialise directly from init overrides (no builders, so the validation paths are exercised)
init_with(; init...) = (p = params_with(quiet_params(); init=(; equilibration_time=0.0, shrink_to_size=false, init...));
                        (M.initialise_sim(p; rng=Xoshiro(1), suppress_prints=true), p))

closest_pair(state, ps) = (as = all_agents(state);
    minimum(M.shortest_distance(a.position, b.position, state.grid_size.dims) / (current_radius(ps, a) + current_radius(ps, b))
            for (i, a) in enumerate(as) for b in as[i+1:end]))

@testset "initialise" begin

    @testset "specify_positions: given positions are used exactly, the rest filled at random" begin
        state, ps = init_with(method="specify_positions", dim_x=1.0, dim_y=1.0,
                              num_OmpA=3, num_OmpCF=0, num_LptD=1, num_BamA=0, num_LPS=5, num_PP=0,
                              positions=positions(OmpA=[(0.2, 0.3)], LPS=[(0.5, 0.5), (0.6, 0.5)]))
        @test agent_counts(state).OmpA == 3 && agent_counts(state).LptD == 1 && agent_counts(state).LPS == 5
        @test state.agents.OMP.OmpA[1].position == SVector(0.2, 0.3)
        @test [l.position for l in state.agents.LPS[1:2]] == pos([(0.5, 0.5), (0.6, 0.5)])
        @test all(all(0 .<= a.position .< state.grid_size.dims) for a in all_agents(state))
        @test consistency_errors(state, ps) == String[]
    end

    @testset "specify_positions rejects invalid position lists" begin
        base = (method="specify_positions", dim_x=1.0, dim_y=1.0, num_OmpA=1, num_OmpCF=0, num_LptD=0,
                num_BamA=0, num_LPS=2, num_PP=0)
        @test_throws AssertionError init_with(; base..., positions=positions(LPS=[(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]))  # too many
        @test_throws AssertionError init_with(; base..., positions=positions(LPS=[(1.5, 0.2)]))                        # outside
        @test_throws AssertionError init_with(; base..., positions=positions(OmpA=[(-0.1, 0.2)]))                      # negative
        @test_throws ErrorException init_with(; base..., positions=nothing)                                           # none at all
    end

    @testset "random: requested agents, inside the domain, assembled as asked" begin
        for assembled in (true, false)
            state, ps = init_with(method="random", dim_x=1.0, dim_y=1.0, num_OmpA=4, num_OmpCF=2, num_LptD=2,
                                  num_BamA=3, num_LPS=30, num_PP=5, complexes_assembled=assembled)
            @test agent_counts(state) == (OmpA=4, OmpCF=2, LptD=2, BamA=3, LPS=30, nascent_OMP=0, nascent_LPS=0)
            @test state.agents.num_PP == 5
            @test all(o.is_tethered == assembled for o in state.agents.OMP.OmpA)
            @test all(l.is_tethered == assembled && l.is_part_of_assembled_complex == assembled for l in state.agents.OMP.LptD)
            @test all(b.is_part_of_assembled_complex == assembled for b in state.agents.OMP.BamA)
            @test consistency_errors(state, ps) == String[]
        end
    end

    @testset "equilibration relieves overlaps" begin
        counts = (method="random", dim_x=1.0, dim_y=1.0, num_OmpA=3, num_OmpCF=2, num_LptD=1, num_BamA=2, num_LPS=80, num_PP=0)
        raw, ps = init_with(; counts..., equilibration_time=0.0)
        settled, ps2 = init_with(; counts..., equilibration_time=0.05)
        @test closest_pair(settled, ps2) > closest_pair(raw, ps)
        @test consistency_errors(settled, ps2) == String[]
    end

    @testset "shrinkage shrinks the domain while holes remain, up to the round limit" begin
        # a sparse membrane keeps its holes through every round, so all rounds are used
        f, rounds = 0.1, 3
        p = params_with(quiet_params(); system=(max_hole_radius=0.05,),
            init=(method="random", dim_x=1.0, dim_y=1.0, num_OmpA=0, num_OmpCF=0, num_LptD=0, num_BamA=0, num_LPS=15,
                  num_PP=0, equilibration_time=0.0, shrink_to_size=true, shrinkage_factor=f,
                  max_num_shrinkage_rounds=rounds, shrinkage_equilibration_time=0.002))
        state = @test_logs (:warn, r"shrinkage rounds") match_mode=:any M.initialise_sim(p; rng=Xoshiro(2), suppress_prints=true)
        @test area(state) ≈ (1 - f)^(2rounds) rtol=1e-9
        @test consistency_errors(state, p) == String[]
    end
end
