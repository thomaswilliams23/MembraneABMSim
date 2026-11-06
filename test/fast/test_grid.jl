# Spatial grid (src/update_grid.jl, src/utils.jl): sync_grid!, rebuild_grid!, build_dense_morton
#
# Contracts: a rebuild leaves every agent in the cell its position says, with structs, flat data
# and grid consistent; cells are numbered by a bijection between cell coordinates and indices;
# the grid follows the domain as it grows. (Whether the search finds the right neighbours is
# tested against brute force in test_neighbour_search.jl.)

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# the coordinate <-> index maps are inverse bijections, in the indexing the grid code uses
# (coords_to_z_ix[(y - 1) * nx + x] for cell (x, y), with x in 1:nx and y in 1:ny)
function morton_is_bijection(nx, ny)
    c2z, z2c = M.build_dense_morton(SVector(nx, ny))
    n = nx * ny
    length(c2z) == n && length(z2c) == 2n || return false
    seen = falses(n)
    for x in 1:nx, y in 1:ny
        z = c2z[(y - 1) * nx + x]
        (1 <= z <= n) && !seen[z] && (z2c[2z-1], z2c[2z]) == (x, y) || return false
        seen[z] = true
    end
    return all(seen)
end

@testset "grid" begin

    @testset "cell numbering is a bijection" begin
        for n in (1, 2, 3, 7, 10, 16, 33)
            @test morton_is_bijection(n, n)
        end
        # non-square domains are not supported yet: the maps use rows and columns inconsistently
        @test_broken morton_is_bijection(4, 7)
        @test_broken morton_is_bijection(9, 2)
    end

    @testset "sync_grid! puts every agent in its own cell" begin
        p = quiet_params(system=(max_steps_between_grid_sync=10^6,))
        for seed in 1:3
            state, ps = random_state(p; OmpA=4, OmpCF=3, LptD=2, BamA=2, LPS=150, L=1.0, rng=Xoshiro(seed))
            t = run_steps!(state, ps, 200)    # agents drift away from their cells (no rebuild)
            M.sync_grid!(state, ps)
            @test all(iszero, moved_since_rebuild(state))
            @test consistency_errors(state, ps) == String[]      # with zero slack: exact cells
            @test state.steps_since_grid_sync == 0
        end
    end

    @testset "the grid follows the domain as it grows and shrinks" begin
        p = quiet_params()
        state, ps = random_state(p; OmpA=3, OmpCF=2, LptD=1, BamA=1, LPS=100, L=1.0, rng=Xoshiro(4))
        for s in (1.3, 1.6, 0.6)
            scale_domain!(state, s)
            M.sync_grid!(state, ps)
            @test consistency_errors(state, ps) == String[]
            @test force_step_matches_oracle(state, ps)
        end
    end

    @testset "growth between rebuilds keeps agents within the search window" begin
        # Known issue: apply_scale_factor! dilates positions and cells but not the distance each
        # agent has moved since the last rebuild, which the neighbour search uses as its margin.
        # After growth an agent can sit further (in cell widths) from its recorded cell than the
        # margin allows. The effect is tiny at real growth rates but exact in principle.
        p = quiet_params(system=(max_steps_between_grid_sync=10^6,))
        state, ps = random_state(p; OmpA=3, OmpCF=2, LptD=1, BamA=1, LPS=120, L=1.0, rng=Xoshiro(6))
        run_steps!(state, ps, 300)                     # drift away from the recorded cells
        scale_domain!(state, 1.3)
        @test_broken consistency_errors(state, ps) == String[]
    end

    @testset "periodic rebuild interval is respected" begin
        for k in (1, 7, 50)
            p = quiet_params(system=(max_steps_between_grid_sync=k,))
            state, ps = random_state(p; LPS=40, L=1.0, rng=Xoshiro(k))
            counts = Int[]
            for i in 1:3(k + 1)                        # a rebuild every k + 1 steps
                run_steps!(state, ps, 1; t0=(i - 1) * ps.system.dt)
                push!(counts, state.steps_since_grid_sync)
            end
            @test maximum(counts) <= k                 # never more than k steps without a rebuild
            @test count(==(0), counts) >= 2            # and it does rebuild
            @test consistency_errors(state, ps) == String[]
        end
    end
end
