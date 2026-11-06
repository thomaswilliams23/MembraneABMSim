# Grid-accelerated force step (src/forces.jl, via the backend's `advance_positions!`) against a
# brute-force oracle that sums the pairwise forces over every pair of agents.
#
# Contract: whatever the grid, cell sizes or sort order, one force step moves every agent
# exactly as the all-pairs sum says it should, including across the periodic boundary, on tiny
# domains with few cells, with insertion partners, and when the grid is stale.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "neighbour search" begin
    p = quiet_params()

    @testset "random packings, $label" for (label, L, n) in (
            ("dense", 1.0, (OmpA=6, OmpCF=4, LptD=2, BamA=2, LPS=250)),
            ("sparse", 2.0, (OmpA=3, OmpCF=2, LptD=1, BamA=1, LPS=20)),
            ("tiny domain, few cells", 0.35, (OmpA=1, OmpCF=1, LptD=1, BamA=0, LPS=6)),
            ("domain one cell wide", 0.15, (OmpA=0, OmpCF=0, LptD=0, BamA=0, LPS=4)))
        for seed in 1:3
            state, ps = random_state(p; n..., L, rng=Xoshiro(seed))
            @test force_step_matches_oracle(state, ps)
            @test consistency_errors(state, ps) == String[]
        end
    end

    @testset "clusters straddling the periodic boundary" begin
        r = p.LPS.radius
        corner = [(mod(x, 1.0), mod(y, 1.0)) for x in -0.1:2r:0.1 for y in -0.1:2r:0.1]
        edge = [(mod(x, 1.0), 0.5) for x in -0.15:1.9r:0.15]
        state, ps = build_state(p; LPS=vcat(corner, edge), OmpCF=[(0.999, 0.25)], LptD=[(0.5, 0.001)])
        @test force_step_matches_oracle(state, ps)
    end

    @testset "stale grid: agents have moved since the last rebuild" begin
        ps = params_with(p; system=(max_steps_between_grid_sync=10^6,))
        for seed in 1:3
            # an overlapping random start makes agents move a long way without a rebuild
            state, ps2 = random_state(ps; OmpA=4, OmpCF=3, LptD=1, BamA=2, LPS=150, L=1.0, rng=Xoshiro(seed))
            t = run_steps!(state, ps2, 300)
            @test maximum(moved_since_rebuild(state)) > 0.1 * ps2.force.sensing_radius
            @test consistency_errors(state, ps2) == String[]
            @test force_step_matches_oracle(state, ps2; t=t + ps2.system.dt)
        end
    end

    @testset "insertion partners feel the spring, not attraction-repulsion" begin
        # Insertions start, but none finishes and nothing is tethered mid-run, so the two known
        # issues (stale tether point after mid-run tethering, overshoot radius after promotion;
        # see consistency.jl) cannot arise and the oracle applies exactly.
        ps = busy_params(OmpA=(tether_rate=0.0,), LPS=(insertion_time=10.0,), insertion=(OMP_assembly_rate=0.01,))
        state, ps2 = random_state(ps; OmpA=3, OmpCF=2, LptD=2, BamA=3, LPS=60, L=1.0, num_PP=3, rng=Xoshiro(4))
        counts0 = agent_counts(state)
        t = step_until!(s -> !isempty(s.agents.nascent.nascent_OMP) && !isempty(s.agents.nascent.nascent_LPS), state, ps2)
        counts = agent_counts(state)
        @test (counts.OmpA, counts.OmpCF, counts.LPS) == (counts0.OmpA, counts0.OmpCF, counts0.LPS)   # nothing promoted
        @test force_step_matches_oracle(state, ps2; t=t + ps2.system.dt)
    end

    @testset "after the domain is rescaled without a rebuild" begin
        state, ps = random_state(p; OmpA=3, OmpCF=2, LptD=1, BamA=1, LPS=120, L=1.0, rng=Xoshiro(9))
        scale_domain!(state, 1.05)
        @test consistency_errors(state, ps) == String[]
        @test force_step_matches_oracle(state, ps)
    end
end
