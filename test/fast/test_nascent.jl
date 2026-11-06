# Nascent agents (src/nascent.jl, src/initialise.jl, src/biology.jl): build_nascent_added_area_lookup,
# compute_nascent_incs and update_nascent_agents! (via whole insertions), promotion to full agents
#
# Contracts: over a full insertion the area lookup adds exactly the new agent's area, never
# removing any; a nascent agent grows steadily from its start size to its final size and is
# then promoted to a full agent of its type, at its position, freeing its inserter.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# follow one nascent OMP (or LPS) from creation to promotion, recording its geometry each step
function track_one_insertion(state, params; lps::Bool=false)
    nascents(s) = lps ? s.agents.nascent.nascent_LPS : s.agents.nascent.nascent_OMP
    t = step_until!(s -> !isempty(nascents(s)), state, params)
    n = only(nascents(state))
    inserter = only(filter(a -> a.index == n.inserting_agent_index, all_agents(state)))
    radii, dists, positions = Float64[], Float64[], SVector{2, Float64}[]
    while !isempty(nascents(state))
        push!(radii, n isa M.NascentOMPAgent ? n.effective_radius : params.LPS.radius)
        push!(dists, n.ideal_dist_from_inserting_agent)
        push!(positions, n.position)
        t = run_steps!(state, params, 1; t0=t)
    end
    promoted = only(filter(a -> a.index == n.index, all_agents(state)))
    return (; n, inserter, radii, dists, positions, promoted, t)
end

@testset "nascent agents" begin

    @testset "area lookup adds exactly one agent's area, never removing any" begin
        for dt in (1e-4, 1e-3), assembly_rate in (0.025, 0.5)
            p = params_with(system=(dt=dt,), insertion=(OMP_assembly_rate=assembly_rate,))
            lookup = M.build_nascent_added_area_lookup(p)
            for type in vcat(collect(OMP_TYPES), "LPS")
                added = getfield(lookup, Symbol(type))
                @test all(added .>= -1e-15)
                @test sum(added) ≈ disk_area(type_radius(p, type)) rtol=1e-3
            end
        end
    end

    @testset "growth increments are all finite and non-negative (no uninitialised entries)" begin
        for rad_or_dist in M.compute_nascent_incs(params_with())
            @test all(x -> isfinite(x) && x >= 0, rad_or_dist)
        end
    end

    @testset "a nascent $type grows steadily and is promoted" for type in OMP_TYPES
        p = single_insertion_params(type)
        state, ps = build_state(p; BamA=[(0.5, 0.5)], LPS=[(0.1, 0.1)], num_PP=1)
        tr = track_one_insertion(state, ps)
        r_full, R_ins = type_radius(ps, type), ps.BamA.radius

        # steady growth up to (within one step's growth of) its final size
        @test all(diff(tr.radii) .>= 0) && all(diff(tr.dists) .> 0)
        dist_step, rad_step = maximum(diff(tr.dists)), maximum(diff(tr.radii); init=0.0)
        @test tr.dists[end] ≈ R_ins + r_full atol=1.01dist_step
        @test tr.radii[end] ≈ r_full atol=1.01rad_step + 1e-12

        # promoted to a full agent of the right type, keeping its identity, inserter freed
        @test agent_type(tr.promoted) == type && !is_nascent(tr.promoted)
        @test tr.promoted.arrival_time == tr.n.arrival_time
        @test tr.inserter.insertion_state == "free"
        @test consistency_errors(state, ps) == String[]
    end

    @testset "a nascent LPS grows steadily and is promoted" begin
        p = quiet_params(LPS=(arrival_rate=1e9, insertion_time=0.02))
        state, ps = build_state(p; LptD=[(0.5, 0.5)], LPS=[(0.1, 0.1)])
        tr = track_one_insertion(state, ps; lps=true)
        @test all(diff(tr.dists) .> 0)
        @test tr.dists[end] ≈ ps.LptD.radius + ps.LPS.radius atol=1.01 * maximum(diff(tr.dists))
        @test tr.promoted isa M.LPSAgent
        @test tr.inserter.insertion_state == "free" && tr.inserter.is_tethered
        @test consistency_errors(state, ps) == String[]
    end

    @testset "known issue: nascent agents overshoot their final size before promotion" begin
        # A nascent agent grows once in the step it is created and once per step after, but is
        # only promoted at the start of the step after its insertion time, so it spends one step
        # one increment past its final size. A promoted large OMP keeps that radius in the flat
        # data until the next grid rebuild (periodic rebuilds are switched off here, so one can
        # never coincide with the promotion and hide the issue).
        p = single_insertion_params("LptD"; system=(max_steps_between_grid_sync=10^6,))
        state, ps = build_state(p; BamA=[(0.5, 0.5)], num_PP=1)
        tr = track_one_insertion(state, ps)
        @test_broken maximum(tr.dists) <= ps.BamA.radius + ps.LptD.radius + 1e-12
        @test_broken maximum(tr.radii) <= ps.LptD.radius + 1e-12
        @test_broken flat_radius(state, tr.promoted) == ps.LptD.radius
    end
end
