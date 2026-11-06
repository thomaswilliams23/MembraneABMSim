# Domain growth (src/update_grid.jl): rescale_domain!, apply_scale_factor!, compute_added_area
#
# Contracts: rescaling is a uniform dilation of the membrane that keeps tethers intact; over one
# complete insertion the domain grows by exactly the new agent's area divided by the density;
# with nothing being inserted the domain does not change.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "domain" begin

    @testset "apply_scale_factor! is a uniform dilation that keeps tethers" begin
        p = quiet_params()
        for s in (1.01, 1.3, 0.97, 0.7)
            state, ps = random_state(p; OmpA=4, OmpCF=2, LptD=2, BamA=2, LPS=40, L=1.0, rng=Xoshiro(1))
            run_steps!(state, ps, 50)                      # tethered agents move off their tether points
            M.sync_grid!(state, ps)                        # (growth on a stale grid: see test_grid.jl)
            as = all_agents(state)
            dims0 = state.grid_size.dims
            pairs0 = [M.shortest_vec(a.position, b.position, dims0) for a in as for b in as]
            tethers0 = [M.shortest_vec(a.position, a.tether_point, dims0) for a in as if is_tethered(a)]

            scale_domain!(state, s)
            dims1 = state.grid_size.dims
            @test dims1 ≈ s * dims0
            @test [M.shortest_vec(a.position, b.position, dims1) for a in as for b in as] ≈ s .* pairs0
            @test [M.shortest_vec(a.position, a.tether_point, dims1) for a in as if is_tethered(a)] ≈ tethers0
            # when shrinking, tether points are wrapped back into the domain
            s < 1 && @test all(all(0 .<= a.tether_point .< dims1) for a in as if is_tethered(a))
            @test consistency_errors(state, ps) == String[]
        end
    end

    @testset "no insertion, no growth" begin
        p = quiet_params()
        state, ps = random_state(p; OmpA=3, LptD=1, BamA=1, LPS=30, L=1.0, rng=Xoshiro(2))
        dims0 = state.grid_size.dims
        run_steps!(state, ps, 200)
        @test state.grid_size.dims == dims0
    end

    @testset "one $type insertion adds the area of one $type" for type in OMP_TYPES
        p = single_insertion_params(type)
        state, ps = build_state(p; BamA=[(0.5, 0.5)], LPS=[(0.1, 0.1), (0.9, 0.2)], num_PP=1)
        area0 = area(state)
        t = step_until!(s -> !isempty(s.agents.nascent.nascent_OMP), state, ps)
        t = step_until!(s -> isempty(s.agents.nascent.nascent_OMP), state, ps; t0=t)
        @test length(getfield(state.agents.OMP, Symbol(type))) == 1 + (type == "BamA")
        @test area(state) - area0 ≈ disk_area(type_radius(ps, type)) / ps.system.density rtol=1e-2
        @test consistency_errors(state, ps) == String[]
    end

    @testset "one LPS insertion adds the area of one LPS" begin
        p = quiet_params(LPS=(arrival_rate=1e9, insertion_time=0.02))
        state, ps = build_state(p; LptD=[(0.5, 0.5)], LPS=[(0.1, 0.1)])
        area0 = area(state)
        t = step_until!(s -> !isempty(s.agents.nascent.nascent_LPS), state, ps)
        t = step_until!(s -> isempty(s.agents.nascent.nascent_LPS), state, ps; t0=t)
        @test length(state.agents.LPS) == 2
        @test area(state) - area0 ≈ disk_area(ps.LPS.radius) / ps.system.density rtol=1e-2
        @test consistency_errors(state, ps) == String[]
    end

    @testset "growth scales with 1 / density" begin
        grown = map((0.5, 1.0)) do density
            p = params_with(single_insertion_params("OmpCF"); system=(density=density,))
            state, ps = build_state(p; BamA=[(0.5, 0.5)], num_PP=1)
            area0 = area(state)
            t = step_until!(s -> !isempty(s.agents.nascent.nascent_OMP), state, ps)
            step_until!(s -> isempty(s.agents.nascent.nascent_OMP), state, ps; t0=t)
            area(state) - area0
        end
        @test grown[1] ≈ 2 * grown[2] rtol=1e-6
    end
end
