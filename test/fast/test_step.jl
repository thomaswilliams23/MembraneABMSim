# One simulation step (src/main.jl, src/sim_state.jl): step!, initialise_sim
#
# Contracts: every step leaves the system consistent; agents are never lost and the membrane
# never shrinks; a step's randomness comes only from the state's RNG (same seed, same result);
# how often the grid is rebuilt does not change the physics.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# a busy membrane: insertions, promotions, tethering and growth all happen within the run
busy_state(; seed=1, sections...) = random_state(busy_params(; sections...);
    OmpA=4, OmpCF=2, LptD=2, BamA=3, LPS=50, L=1.0, num_PP=4, rng=Xoshiro(seed))

@testset "step" begin

    @testset "initialise_sim gives a consistent state" begin
        for seed in 1:3
            state, ps = busy_state(; seed)
            @test consistency_errors(state, ps) == String[]
            @test state.backend isa M.SimBackendCPU
            @test state.steps_since_grid_sync == 0
        end
    end

    @testset "consistent after every step; agents never lost; membrane never shrinks" begin
        state, ps = busy_state(seed=2)
        n_inconsistent, lost, shrank = 0, false, false
        n_prev, area_prev = state.grid_size.num_agents, area(state)
        t = 0.0
        for _ in 1:2000
            t = run_steps!(state, ps, 1; t0=t)
            n_inconsistent += !isempty(consistency_errors(state, ps))
            lost |= state.grid_size.num_agents < n_prev
            shrank |= area(state) < area_prev
            n_prev, area_prev = state.grid_size.num_agents, area(state)
        end
        @test n_inconsistent == 0
        @test !lost && !shrank
        @test state.grid_size.num_agents > 50 + 4 + 2 + 2 + 3     # it was busy (effectively certain, see busy_params)
    end

    @testset "same seed, same result; different seed, different result" begin
        simulate(seed) = ((s, ps) = busy_state(; seed); run_steps!(s, ps, 1500); s)
        a, b, c = simulate(5), simulate(5), simulate(6)
        @test deep_equal(a.agents, b.agents) && a.grid_size.dims == b.grid_size.dims
        @test !deep_equal(a.agents, c.agents)

        # the global RNG plays no part once the state has its own
        s1, ps = busy_state(seed=7)
        s2, _ = busy_state(seed=7)
        Random.seed!(1); run_steps!(s1, ps, 500)
        Random.seed!(2); run_steps!(s2, ps, 500)
        @test deep_equal(s1.agents, s2.agents)
    end

    @testset "with diffusion: consistent and reproducible" begin
        p = busy_params(force=(temperature=1e-3, diffusion_mode="LPS_only"))
        simulate(seed) = ((s, ps) = random_state(p; OmpA=3, LptD=2, BamA=2, LPS=40, L=1.0, num_PP=2, rng=Xoshiro(seed));
                          run_steps!(s, ps, 800); (s, ps))
        (a, ps), (b, _) = simulate(3), simulate(3)
        @test consistency_errors(a, ps) == String[]
        @test deep_equal(a.agents, b.agents)
    end

    @testset "grid rebuild interval does not change the physics" begin
        # with no randomness after initialisation, rebuilding every step or every 100 steps gives
        # the same trajectory up to floating-point summation order. Rounding differences grow
        # chaotically from an overlapping start (about 1e-16 after 100 steps, 1e-15 after 300,
        # 1e-8 after 1000), so compare just past the default run's first rebuild, with a wide
        # margin. Each force step's correctness on a stale grid is checked in test_neighbour_search.jl.
        finals = map((1, 100)) do k
            p = quiet_params(system=(max_steps_between_grid_sync=k,))
            state, ps = random_state(p; OmpA=3, OmpCF=2, LptD=1, BamA=2, LPS=80, L=1.0, rng=Xoshiro(8))
            run_steps!(state, ps, 150)
            positions_by_index(state), state.grid_size.dims
        end
        (pos1, dims), (pos2, _) = finals
        @test all(M.shortest_distance(pos1[i], pos2[i], dims) < 1e-9 for i in keys(pos1))
    end
end
