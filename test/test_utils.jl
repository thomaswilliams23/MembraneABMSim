# Unit tests for the helpers in src/utils.jl, in source order:
#   shortest_vec, shortest_distance, build_dense_morton, exposed_area, change_field,
#   make_identifier, parse_identifier, is_interval_step, copy_data_to_cpu_yn,
#   fast_floor_int32, get_centroid

include(joinpath(@__DIR__, "setup.jl"))


@testset "shortest_vec / shortest_distance (periodic minimum image)" begin
    dims = SVector(1.0, 2.0)

    # simple interior case: no wrapping
    a = SVector(0.2, 0.5)
    b = SVector(0.3, 0.9)
    @test M.shortest_vec(a, b, dims) ≈ SVector(0.1, 0.4)
    @test M.shortest_distance(a, b, dims) ≈ norm(SVector(0.1, 0.4))

    # straddling the boundary: shortest path goes through the edge
    a = SVector(0.05, 0.1)
    b = SVector(0.95, 1.9)
    @test M.shortest_vec(a, b, dims) ≈ SVector(-0.1, -0.2)
    @test M.shortest_distance(a, b, dims) ≈ norm(SVector(0.1, 0.2))

    # property checks on random pairs
    rng_points = [SVector(rand() * dims[1], rand() * dims[2]) for _ in 1:200]
    for i in 1:2:length(rng_points)
        p, q = rng_points[i], rng_points[i+1]
        v = M.shortest_vec(p, q, dims)

        # each component within half a period
        @test all(abs.(v) .<= dims ./ 2 .+ 1e-12)

        # antisymmetry
        @test M.shortest_vec(q, p, dims) ≈ -v atol=1e-12

        # invariance under shifting either point by whole periods
        shift = SVector(3 * dims[1], -2 * dims[2])
        @test M.shortest_vec(p + shift, q, dims) ≈ v atol=1e-12
        @test M.shortest_vec(p, q - shift, dims) ≈ v atol=1e-12

        # distance is symmetric and bounded by half the domain diagonal
        @test M.shortest_distance(p, q, dims) ≈ M.shortest_distance(q, p, dims)
        @test M.shortest_distance(p, q, dims) <= norm(dims ./ 2) + 1e-12
    end
end


@testset "build_dense_morton bijection" begin
    # check the mapping using the indexing convention of the callers
    # (src/update_grid.jl, src/forces.jl, src/biology.jl): ix = (y-1)*num_cells[1] + x,
    # with x ∈ 1:num_cells[1], y ∈ 1:num_cells[2]
    function morton_consistent(num_cells)
        coords_to_z, z_to_coords = M.build_dense_morton(num_cells)
        n = prod(num_cells)
        sort(coords_to_z) == collect(1:n) || return false
        for x in 1:num_cells[1], y in 1:num_cells[2]
            z = coords_to_z[(y-1)*num_cells[1] + x]
            (z_to_coords[2z-1], z_to_coords[2z]) == (x, y) || return false
        end
        return true
    end

    for n in (1, 2, 5, 8, 13)
        @test morton_consistent(SVector(n, n))
    end

    # Non-square grids: build_dense_morton strides by num_cells[2] but callers stride by
    # num_cells[1]. Known issue (docs warn dim_x != dim_y is unsupported).
    @test_broken morton_consistent(SVector(4, 2))
end


@testset "exposed_area" begin
    # brute-force estimate: area of the substrate disc not covered by the inserting disc
    # (inserting agent at the origin, substrate at (dist, 0))
    function exposed_area_grid(rad_ins, rad_sub, dist; n=1500)
        h = 2 * rad_sub / n
        covered = 0
        for i in 1:n, j in 1:n
            x = dist - rad_sub + (i - 0.5) * h
            y = -rad_sub + (j - 0.5) * h
            in_sub = (x - dist)^2 + y^2 <= rad_sub^2
            in_ins = x^2 + y^2 <= rad_ins^2
            covered += in_sub && !in_ins
        end
        return covered * h^2
    end

    # (inserting radius, substrate radius): smaller, equal and larger substrate
    for (r_ins, r_sub) in ((0.02, 0.01), (0.02, 0.02), (0.02, 0.03))
        full = π * r_sub^2

        # (near-)zero separation
        @test M.exposed_area(r_ins, r_sub, 0.0) == 0.0
        @test M.exposed_area(r_ins, r_sub, 1e-7) == 0.0

        # no overlap: whole substrate exposed
        for d in (r_ins + r_sub, r_ins + r_sub + 1e-3, 3 * (r_ins + r_sub))
            @test M.exposed_area(r_ins, r_sub, d) ≈ full
        end

        # bounded and non-decreasing as the substrate moves out
        ds = range(1e-5, r_ins + r_sub + 0.01; length=400)
        areas = [M.exposed_area(r_ins, r_sub, d) for d in ds]
        @test all(-1e-15 .<= areas .<= full + 1e-15)
        @test all(diff(areas) .>= -1e-15)

        # matches the geometric definition at partial overlaps
        for frac in (0.3, 0.6, 0.9)
            d = abs(r_ins - r_sub) + frac * (r_ins + r_sub - abs(r_ins - r_sub))
            @test M.exposed_area(r_ins, r_sub, d) ≈ exposed_area_grid(r_ins, r_sub, d) atol=5e-3*full
        end
    end

    # substrate entirely inside the inserting agent: nothing exposed
    @test M.exposed_area(0.03, 0.01, 0.015) ≈ 0.0 atol=1e-15

    # inserting agent entirely inside the substrate: only the annulus is exposed
    @test M.exposed_area(0.01, 0.03, 0.015) ≈ π * (0.03^2 - 0.01^2)
end


# struct definitions must be at top level, not inside a @testset
struct ChangeFieldTestStruct
    a::Int
    b::String
    c::Float64
end

@testset "change_field" begin
    obj = ChangeFieldTestStruct(1, "x", 2.5)

    new_obj = M.change_field(obj, "b", "y")
    @test new_obj isa ChangeFieldTestStruct
    @test (new_obj.a, new_obj.b, new_obj.c) == (1, "y", 2.5)
    @test (obj.a, obj.b, obj.c) == (1, "x", 2.5)          # original untouched

    # unknown field
    @test_throws ErrorException M.change_field(obj, "d", 0)

    # nested replacement on real simulation parameters
    params = M.parse_config(DEMO_CONFIG)
    new_system = M.change_field(params.system, "dt", 0.123)
    new_params = M.change_field(params, "system", new_system)
    @test new_params.system.dt == 0.123
    @test new_params.system.vis_dt == params.system.vis_dt
    @test new_params.force === params.force
    @test params.system.dt != 0.123
end


@testset "make_identifier / parse_identifier round trip" begin
    agent_types = ["OmpA", "OmpCF", "BamA", "LptD", "LPS"]

    # (is_inserting, is_nascent, nascent_ix) combinations that occur in the code:
    # plain agents carry nascent_ix=0; inserting/nascent agents use nascent_ix ≥ 1
    # (see update_flat_data_nascent_agents!, src/biology.jl)
    states = [(false, false, 0)]
    for ix in (1, 2, 9, 10, 11, 50, 98)
        push!(states, (true, false, ix))
        push!(states, (false, true, ix))
    end

    for agent_type in agent_types, is_tethered in (false, true), (is_inserting, is_nascent, ix) in states
        id = M.make_identifier(; is_tethered, agent_type, is_inserting, is_nascent, nascent_ix=ix)
        parsed = M.parse_identifier(id)
        @test parsed == (is_tethered, agent_type, is_nascent, is_inserting, ix)

        # parity encodes OMP (odd) vs LPS (even); force code relies on `id & 1`
        # (src/forces.jl, mu_attr selection), including for negative identifiers
        @test (id & 1) == (agent_type == "LPS" ? 0 : 1)
    end

    @test_throws AssertionError M.make_identifier(; is_tethered=false, agent_type="LPS",
                                                   is_inserting=false, is_nascent=true, nascent_ix=99)
    @test_throws AssertionError M.make_identifier(; is_tethered=false, agent_type="BamA",
                                                   is_inserting=true, is_nascent=true, nascent_ix=1)
end


@testset "is_interval_step" begin
    # step indices (0-based) at which the check fires, for t built either exactly or by accumulation
    function hit_steps(interval, dt, n_steps; accumulate=false)
        hits = Int[]
        t = 0.0
        for k in 0:n_steps
            t_k = accumulate ? t : k * dt
            M.is_interval_step(t_k, interval, dt) && push!(hits, k)
            t += dt
        end
        return hits
    end

    for (interval, dt) in ((0.1, 1e-4), (0.25, 0.01), (20.0, 0.01), (1.0, 0.1))
        k_per = round(Int, interval / dt)
        n_steps = 5 * k_per
        expected = collect(0:k_per:n_steps)                  # includes t = 0
        @test hit_steps(interval, dt, n_steps) == expected
        @test hit_steps(interval, dt, n_steps; accumulate=true) == expected
    end

    # interval not a whole number of steps: rounded to the nearest, still one hit per period
    @test hit_steps(0.104, 0.01, 50) == collect(0:10:50)
    @test hit_steps(0.106, 0.01, 55) == collect(0:11:55)

    # interval shorter than a step: fires every step
    @test hit_steps(0.001, 0.01, 20) == collect(0:20)
end


@testset "copy_data_to_cpu_yn" begin
    # parameters with easily distinguishable time scales
    function make_params(; dt, attempt_dt, vis_dt)
        p = M.parse_config(DEMO_CONFIG)
        system = M.change_field(M.change_field(p.system, "dt", dt), "vis_dt", vis_dt)
        insertion = M.change_field(p.insertion, "attempt_dt", attempt_dt)
        return M.change_field(M.change_field(p, "system", system), "insertion", insertion)
    end
    params = make_params(dt=0.01, attempt_dt=0.1, vis_dt=0.25)

    BamA(state) = M.BamAAgent(1, SVector(0.5, 0.5), 0.0, false, state)
    no_BamA = M.BamAAgent[]
    stalled = [BamA("free"), BamA("stalled")]
    not_stalled = [BamA("free"), BamA("free")]

    # next step is a visualisation time: always copy
    @test M.copy_data_to_cpu_yn(no_BamA, params, 0.24)
    @test M.copy_data_to_cpu_yn(not_stalled, params, 0.24)

    # next step is an insertion attempt time (not a vis time): copy only if a BamA is stalled
    @test M.copy_data_to_cpu_yn(stalled, params, 0.29)
    @test !M.copy_data_to_cpu_yn(not_stalled, params, 0.29)
    @test !M.copy_data_to_cpu_yn(no_BamA, params, 0.29)

    # next step is neither: never copy
    @test !M.copy_data_to_cpu_yn(stalled, params, 0.31)

    # Step alignment over a run, with t accumulated as t += dt (so floating-point drift is
    # included). A copy must happen on exactly the step before each output step, using
    # run_sim's convention mod(time_ix, round(vis_dt/dt)) == 0, and, if a BamA is stalled,
    # also on the step before each insertion attempt step.
    function copy_steps(agents, params, n_steps)
        steps = Int[]
        t = 0.0
        for k in 0:n_steps
            M.copy_data_to_cpu_yn(agents, params, t) && push!(steps, k)
            t += params.system.dt
        end
        return steps
    end
    function expected_steps(params, n_steps; with_attempts)
        dt = params.system.dt
        k_vis = round(Int, params.system.vis_dt / dt)
        k_att = round(Int, params.insertion.attempt_dt / dt)
        return [k for k in 0:n_steps
                if mod(k + 1, k_vis) == 0 || (with_attempts && mod(k + 1, k_att) == 0)]
    end

    long_vis = make_params(dt=0.01, attempt_dt=0.1, vis_dt=20.0)
    for p in (params, long_vis)
        n_steps = 3 * round(Int, p.system.vis_dt / p.system.dt)
        @test copy_steps(no_BamA, p, n_steps) == expected_steps(p, n_steps; with_attempts=false)
        @test copy_steps(not_stalled, p, n_steps) == expected_steps(p, n_steps; with_attempts=false)
        @test copy_steps(stalled, p, n_steps) == expected_steps(p, n_steps; with_attempts=true)
    end

    # exactly one copy per vis time, even when vis_dt ≫ dt
    @test length(copy_steps(no_BamA, long_vis, 4000)) == 2
end


@testset "fast_floor_int32" begin
    check(x) = (y = M.fast_floor_int32(x); y isa Int32 && y == floor(Int32, x))

    # exact integers, and values just either side of them
    for k in Float32.(-1000:1000)
        @test check(k)
        @test check(prevfloat(k))
        @test check(nextfloat(k))
    end

    # random values
    for _ in 1:2000
        @test check(Float32((2 * rand() - 1) * 1e6))
        @test check(Float32(2 * rand() - 1))
    end
end


@testset "get_centroid" begin
    dims = SVector(1.0, 2.0)

    # single point
    @test M.get_centroid([SVector(0.3, 0.4)], dims) == SVector(0.3, 0.4)

    # interior cluster: plain mean
    pts = [SVector(0.2, 0.5), SVector(0.4, 0.7), SVector(0.3, 0.9)]
    @test M.get_centroid(pts, dims) ≈ SVector(0.3, 0.7)

    # cluster straddling the corner: centroid sits on the corner
    pts = [SVector(0.95, 1.9), SVector(0.05, 0.1), SVector(0.95, 0.1), SVector(0.05, 1.9)]
    @test M.shortest_distance(M.get_centroid(pts, dims), SVector(0.0, 0.0), dims) < 1e-12

    # cluster straddling the x boundary only
    pts = [SVector(0.9, 1.0), SVector(0.2, 1.2)]
    @test M.shortest_distance(M.get_centroid(pts, dims), SVector(0.05, 1.1), dims) < 1e-12

    # random compact clusters: in-domain, translation-equivariant and order-independent
    for _ in 1:100
        centre = SVector(rand() * dims[1], rand() * dims[2])
        pts = [mod.(centre + (rand(2) .- 0.5) .* dims ./ 4, dims) for _ in 1:rand(2:10)]
        pts = SVector{2, Float64}.(pts)
        c = M.get_centroid(pts, dims)

        @test all(0 .<= c .< dims)

        shift = SVector(rand() * dims[1], rand() * dims[2])
        c_shifted = M.get_centroid([mod.(p + shift, dims) for p in pts], dims)
        @test M.shortest_distance(c_shifted, mod.(c + shift, dims), dims) < 1e-12

        @test M.shortest_distance(M.get_centroid(reverse(pts), dims), c, dims) < 1e-12
    end
end
