# Hole detection and growth blocking (src/update_grid.jl): membrane_contains_hole,
# and the allow_membrane_growth decision made at each grid rebuild
#
# Contracts: membrane_contains_hole agrees with a brute-force largest-empty-disk search on
# clear-cut cases (a hole well above the threshold, or none anywhere near it); it is off when
# no threshold is set; and the domain does not grow while the membrane has a hole.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# LPS on a hexagonal lattice tiling an L x L domain (touching neighbours), minus any lattice
# sites inside a disk of radius `hole` at the centre
function lps_lattice(r, L; hole=0.0)
    nx = floor(Int, L / 2r)
    ny = floor(Int, L / (sqrt(3) * r))
    ny -= isodd(ny)                         # even row count keeps the lattice periodic in y
    dx, dy = L / nx, L / ny
    pts = [(mod((i + 0.5 * isodd(j)) * dx, L), j * dy) for i in 0:nx-1 for j in 0:ny-1]
    c = (L / 2, L / 2)
    return filter(q -> hypot(q[1] - c[1], q[2] - c[2]) >= hole, pts)
end

@testset "holes" begin
    r = quiet_params().LPS.radius
    L = 0.8

    @testset "agrees with brute force on clear-cut cases" begin
        for hole in (0.0, 0.1, 0.2)
            state, ps = build_state(quiet_params(); LPS=lps_lattice(r, L; hole), L)
            free = largest_free_disk(state, ps; spacing=0.002)
            for threshold in (0.25free, 0.5free, 2free, 4free)
                threshold > 0.005 || continue              # too small for the detector's pixels
                threshold < L / 4 || continue              # holes are small relative to the domain
                p_t = params_with(ps; system=(max_hole_radius=threshold,))
                found = M.membrane_contains_hole(state.agents, state.grid_size, p_t)
                threshold <= 0.5free && @test found        # a disk twice the threshold fits
                threshold >= 2free && @test !found         # nothing even half the threshold fits
            end
        end
    end

    @testset "off when no threshold is set" begin
        state, ps = build_state(quiet_params(); LPS=lps_lattice(r, L; hole=0.25), L)
        @test isnothing(ps.system.max_hole_radius)
        @test !M.membrane_contains_hole(state.agents, state.grid_size, ps)
    end

    @testset "the domain does not grow while the membrane has a hole" begin
        # a sparse membrane (plenty of room for holes) with an LptD inserting LPS: it grows only
        # when hole checking is off
        for max_hole_radius in (nothing, 0.05)
            p = quiet_params(LPS=(arrival_rate=1e9, insertion_time=0.01), system=(max_hole_radius=max_hole_radius,))
            state, ps = build_state(p; LptD=[(0.4, 0.4)], LPS=[(0.1, 0.1), (0.7, 0.2)], L)
            area0 = area(state)
            run_steps!(state, ps, 300)                   # several LPS insertions, with grid rebuilds
            @test length(state.agents.LPS) > 2
            @test state.allow_membrane_growth == isnothing(max_hole_radius)
            @test (area(state) > area0) == isnothing(max_hole_radius)
        end
    end
end
