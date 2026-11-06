# Periodic geometry and small numeric utilities (src/utils.jl):
#   shortest_vec, shortest_distance, get_centroid, exposed_area, fast_floor_int32
#
# Contracts: minimum-image vectors on a periodic domain, centroids that respect the wrap, and
# the exposed (non-overlapped) area of a disk, checked through its limits and monotonicity.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

const GEOM_RNG = Xoshiro(20)
rand_point(dims) = dims .* SVector(rand(GEOM_RNG), rand(GEOM_RNG))

@testset "geometry" begin

    @testset "shortest_vec / shortest_distance: minimum image" begin
        for dims in (SVector(1.0, 1.0), SVector(2.5, 0.7))
            for _ in 1:500
                p, q = rand_point(dims), rand_point(dims)
                v = M.shortest_vec(p, q, dims)

                # points from p to (an image of) q, and is the shortest such vector
                @test all(abs.(v) .<= dims ./ 2 .+ 1e-12)
                periods = (p + v - q) ./ dims             # p + v lands on q up to whole periods
                @test all(isapprox.(periods, round.(periods); atol=1e-9))

                # antisymmetric, distance is its length and symmetric
                @test M.shortest_vec(q, p, dims) ≈ -v atol=1e-12
                @test M.shortest_distance(p, q, dims) ≈ norm(v)
                @test M.shortest_distance(p, q, dims) ≈ M.shortest_distance(q, p, dims)

                # invariant under shifting either point by whole periods, or both by anything
                k = SVector(rand(GEOM_RNG, -2:2), rand(GEOM_RNG, -2:2))
                @test M.shortest_vec(p, q + k .* dims, dims) ≈ v atol=1e-9
                shift = rand_point(dims)
                @test M.shortest_vec(mod.(p + shift, dims), mod.(q + shift, dims), dims) ≈ v atol=1e-9
            end
            # a point is at distance zero from itself and its images
            p = rand_point(dims)
            @test M.shortest_distance(p, p, dims) == 0
            @test M.shortest_distance(p, p + dims, dims) ≈ 0 atol=1e-12
        end
    end

    @testset "get_centroid: respects the periodic wrap" begin
        dims = SVector(1.0, 1.0)
        p = SVector(0.3, 0.6)
        @test M.get_centroid([p], dims) == p

        # a symmetric cluster straddling the corner has its centroid at the corner, not the middle
        cluster = [SVector(0.02, 0.02), SVector(0.98, 0.02), SVector(0.02, 0.98), SVector(0.98, 0.98)]
        c = M.get_centroid(cluster, dims)
        @test M.shortest_distance(c, SVector(0.0, 0.0), dims) < 1e-12

        # translating a cluster translates its centroid, whatever the wrap
        for _ in 1:100
            centre = rand_point(dims)
            pts = [mod.(centre + 0.1 .* SVector(randn(GEOM_RNG), randn(GEOM_RNG)), dims) for _ in 1:8]
            shift = rand_point(dims)
            c1 = M.get_centroid(pts, dims)
            c2 = M.get_centroid([mod.(q + shift, dims) for q in pts], dims)
            @test M.shortest_distance(mod.(c1 + shift, dims), c2, dims) < 1e-9
            @test all(0 .<= c2 .< dims)
        end
    end

    @testset "exposed_area: limits, bounds and monotonicity" begin
        for (r_ins, r_sub) in ((0.17, 0.05), (0.17, 0.17), (0.26, 0.035), (0.1, 0.2))
            full = disk_area(r_sub)

            # coincident centres: nothing exposed if the substrate fits inside the inserter, else
            # the ring outside it. (Known issue: below dist 1e-6 exposed_area returns 0 even for a
            # larger substrate. The model never hits this case: substrates start inside BamA, or
            # exactly coincident with an equal radius.)
            if r_sub <= r_ins
                @test M.exposed_area(r_ins, r_sub, 0.0) ≈ 0 atol=1e-15
            else
                @test_broken M.exposed_area(r_ins, r_sub, 0.0) ≈ disk_area(r_sub) - disk_area(r_ins)
            end
            @test M.exposed_area(r_ins, r_sub, r_ins + r_sub) ≈ full rtol=1e-9
            @test M.exposed_area(r_ins, r_sub, 2 * (r_ins + r_sub)) ≈ full rtol=1e-9

            # substrate entirely inside the inserter: nothing exposed
            if r_sub < r_ins
                @test M.exposed_area(r_ins, r_sub, 0.5 * (r_ins - r_sub)) ≈ 0 atol=1e-12
            end

            # within [0, full], non-decreasing and continuous as the substrate moves out
            ds = range(1e-5, r_ins + r_sub; length=2001)
            a = [M.exposed_area(r_ins, r_sub, d) for d in ds]
            @test all(-1e-12 .<= a .<= full + 1e-12)
            @test all(diff(a) .>= -1e-12)
            @test maximum(abs, diff(a)) < 0.01 * full
        end
    end

    @testset "fast_floor_int32 agrees with floor" begin
        xs = Float32[-3.5, -3.0, -0.5, -0.0, 0.0, 0.3, 1.0, 2.999, 1e6, -1e6]
        append!(xs, Float32.(200 .* (rand(GEOM_RNG, 1000) .- 0.5)))
        @test all(M.fast_floor_int32(x) == Int32(floor(x)) for x in xs)
    end
end
