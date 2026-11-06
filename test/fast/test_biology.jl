# Insertion biology (src/biology.jl): update_BAM_subsystem!, update_Lpt_subsystem!,
# update_tethering_and_assembly! (driven through step!)
#
# Rates are forced to 0 or effectively infinity so each process is deterministic. Contracts:
# only legal state transitions; each OMP insertion consumes one polypeptide; only assembled
# BamA/LptD insert; insertion attempts happen only on attempt steps; lipid-dependent insertion
# needs an LPS within sensing range; tethering and assembly happen at once at rate infinity and
# never at rate 0; the inserted OMP type follows the insertion probabilities. The state stays
# consistent throughout.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

const INF_RATE = 1e9

# insertion states of every BamA and LptD, keyed by agent index
insertion_states(state) = Dict(a.index => a.insertion_state for a in vcat(state.agents.OMP.BamA, state.agents.OMP.LptD))

const LEGAL_BamA = Dict("free" => ("free", "stalled", "embedding"),     # binding and inserting can share a step
                        "stalled" => ("stalled", "embedding"),
                        "embedding" => ("embedding", "free"))
const LEGAL_LptD = Dict("free" => ("free", "embedding"), "embedding" => ("embedding", "free"))

@testset "biology" begin

    @testset "busy membrane: legal transitions, polypeptides conserved, consistent" begin
        p = busy_params(insertion=(PP_arrival_rate=0.0,))
        state, ps = random_state(p; OmpA=4, OmpCF=2, LptD=3, BamA=4, LPS=50, L=1.0, num_PP=12, rng=Xoshiro(7))
        n_OMP0 = sum(length(getfield(state.agents.OMP, Symbol(t))) for t in OMP_TYPES)
        illegal, inconsistent = String[], 0
        t = 0.0
        for _ in 1:3000
            before = insertion_states(state)
            t = run_steps!(state, ps, 1; t0=t)
            for a in vcat(state.agents.OMP.BamA, state.agents.OMP.LptD)
                haskey(before, a.index) || continue           # inserted this step
                legal = a isa M.BamAAgent ? LEGAL_BamA : LEGAL_LptD
                a.insertion_state in legal[before[a.index]] ||
                    push!(illegal, "$(agent_type(a)) $(a.index): $(before[a.index]) -> $(a.insertion_state)")
            end
            inconsistent += !isempty(consistency_errors(state, ps))
        end
        @test illegal == String[]
        @test inconsistent == 0

        # every polypeptide used went into exactly one OMP (inserted, or still being inserted)
        n_OMP = sum(length(getfield(state.agents.OMP, Symbol(t))) for t in OMP_TYPES)
        in_progress = count(b -> b.insertion_state in ("stalled", "embedding"), state.agents.OMP.BamA)
        @test 12 - state.agents.num_PP == (n_OMP - n_OMP0) + in_progress
        # something happened (busy_params makes this effectively certain: P(no LPS) < e^-20)
        @test n_OMP > n_OMP0 && length(state.agents.LPS) > 50
    end

    @testset "polypeptide binding needs an assembled BamA and a positive rate" begin
        for (assembled, bind_rate, should_bind) in ((true, INF_RATE, true), (false, INF_RATE, false), (true, 0.0, false))
            p = single_insertion_params("OmpA"; insertion=(PP_bind_rate=bind_rate, attempt_dt=1.0))
            state, ps = build_state(p; BamA=[(0.3, 0.3), (0.7, 0.7)], num_PP=5, assembled)
            run_steps!(state, ps, 5)
            n_bound = count(b -> b.insertion_state == "stalled", state.agents.OMP.BamA)
            @test n_bound == (should_bind ? 2 : 0)
            @test state.agents.num_PP == 5 - n_bound
        end
    end

    @testset "polypeptides arrive at the given rate (0 means never)" begin
        p = quiet_params(insertion=(PP_arrival_rate=0.0,))
        state, ps = build_state(p; LPS=[(0.5, 0.5)], num_PP=3)
        run_steps!(state, ps, 500)
        @test state.agents.num_PP == 3
        p = quiet_params(insertion=(PP_arrival_rate=INF_RATE,))
        state, ps = build_state(p; LPS=[(0.5, 0.5)])
        run_steps!(state, ps, 1)
        @test state.agents.num_PP > 0
    end

    @testset "insertion attempts happen only on attempt steps" begin
        p = single_insertion_params("OmpA"; insertion=(attempt_dt=0.0123,))
        state, ps = build_state(p; BamA=[(0.5, 0.5)], num_PP=1)
        t = step_until!(s -> !isempty(s.agents.nascent.nascent_OMP), state, ps)
        @test M.is_interval_step(t, ps.insertion.attempt_dt, ps.system.dt)
        @test only(state.agents.nascent.nascent_OMP).arrival_time ≈ t
    end

    @testset "lipid-dependent insertion needs an LPS within sensing range" begin
        p = single_insertion_params("OmpA"; insertion=(type="lipid-dependent", attempt_dt=1e-3))
        R, r, sens = p.BamA.radius, p.LPS.radius, p.force.sensing_radius
        reach = R + sens + r
        for (lps_distance, should_insert) in ((0.5reach, true), (0.95reach, true), (1.3reach, false))
            state, ps = build_state(p; BamA=[(0.5, 0.5)], LPS=[(0.5 + lps_distance, 0.5)], num_PP=1, L=2.0)
            run_steps!(state, ps, 50)
            @test num_nascent(state) + length(state.agents.OMP.OmpA) == Int(should_insert)
            @test only(state.agents.OMP.BamA).insertion_state == (should_insert ? "embedding" : "stalled")
        end
        # with no LPS at all the BamA stays stalled
        state, ps = build_state(p; BamA=[(0.5, 0.5)], OmpCF=[(0.6, 0.8)], num_PP=1)
        run_steps!(state, ps, 200)
        @test only(state.agents.OMP.BamA).insertion_state == "stalled"
        @test num_nascent(state) == 0
    end

    @testset "inserted OMP type follows one-hot insertion probabilities" begin
        for type in OMP_TYPES
            p = single_insertion_params(type)
            state, ps = build_state(p; BamA=[(0.5, 0.5)], num_PP=1)
            step_until!(s -> !isempty(s.agents.nascent.nascent_OMP), state, ps)
            @test only(state.agents.nascent.nascent_OMP).OMP_type == type
        end
    end

    @testset "LPS insertion needs an assembled LptD and a positive arrival rate" begin
        for (assembled, rate, should_insert) in ((true, INF_RATE, true), (false, INF_RATE, false), (true, 0.0, false))
            p = quiet_params(LPS=(arrival_rate=rate,))
            state, ps = build_state(p; LptD=[(0.3, 0.3), (0.7, 0.7)], assembled)
            run_steps!(state, ps, 3)
            @test length(state.agents.nascent.nascent_LPS) == (should_insert ? 2 : 0)
            @test all(l.insertion_state == (should_insert ? "embedding" : "free") for l in state.agents.OMP.LptD)
            @test consistency_errors(state, ps) == String[]
        end
    end

    @testset "tethering and assembly happen at once at rate infinity, never at rate 0" begin
        for rate in (INF_RATE, 0.0)
            p = quiet_params(OmpA=(tether_rate=rate,), BamA=(complex_assembly_rate=rate,), LptD=(complex_assembly_rate=rate,))
            state, ps = build_state(p; OmpA=[(0.2, 0.2), (0.8, 0.3)], BamA=[(0.5, 0.8)], LptD=[(0.3, 0.6)],
                                    LPS=[(0.6, 0.5)], assembled=false)
            positions = positions_by_index(state)
            run_steps!(state, ps, 1)
            on = rate > 0
            @test all(o.is_tethered == on for o in state.agents.OMP.OmpA)
            @test all(b.is_part_of_assembled_complex == on for b in state.agents.OMP.BamA)
            @test all(l.is_part_of_assembled_complex == on && l.is_tethered == on for l in state.agents.OMP.LptD)
            # tethered where they stood when the tether formed
            for a in all_agents(state)
                is_tethered(a) && @test a.tether_point == positions[a.index]
            end
            @test consistency_errors(state, ps) == String[]
        end
    end

    @testset "known issue: a newly tethered agent is frozen on the CPU until the next rebuild" begin
        # update_tethering_and_assembly! sets the tether point in the structs but not in the flat
        # data, which the CPU force step uses for the tether constraint. Until the next grid
        # rebuild the constraint is checked against the stale point (0, 0), so every move is
        # rejected. (The GPU backends set it on the device.)
        p = quiet_params(OmpA=(tether_rate=INF_RATE,))
        r_pair = p.OmpA.radius + p.LPS.radius
        state, ps = build_state(p; OmpA=[(0.5, 0.5)], LPS=[(0.5 + r_pair + 0.03, 0.5)], assembled=false)  # the LPS attracts the OmpA
        a = only(state.agents.OMP.OmpA)
        t = run_steps!(state, ps, 1)
        @test a.is_tethered
        @test_broken flat_tether(state, a) == a.tether_point
        p0 = a.position
        run_steps!(state, ps, 20; t0=t)
        @test_broken a.position != p0
    end
end
