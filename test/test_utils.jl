# Unit tests for pure helpers in src/utils.jl

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
