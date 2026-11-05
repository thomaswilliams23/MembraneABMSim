# Unit tests for the pairwise attraction-repulsion force (src/forces.jl)

# Representative parameters (cf. config/simulation/demo.json)
const FP = (
    dims = SVector(1.0, 1.0),
    r_a = 0.02,
    r_b = 0.015,
    sensing_radius = 0.02,
    mu_attr = 0.1,
    mu_rep = 1.0,
    max_repulsion = -0.2,
    rho = 0.5,
    k_C = 3.0,
)

# attracttion-repulsion force on agent at `p` due to neighbour at `q`
attr_rep_force(p, q; mu_attr=FP.mu_attr, dims=FP.dims) = M.compute_attr_rep_force(
    p, FP.r_a, q, FP.r_b, dims, FP.sensing_radius,
    mu_attr, FP.mu_rep, FP.max_repulsion, FP.rho, FP.k_C)

# signed magnitude along the separation direction (positive = attraction)
function signed_mag(dist; mu_attr=FP.mu_attr, dir=SVector(1.0, 0.0))
    p = SVector(0.5, 0.5)
    F = attr_rep_force(p, p + dist * dir; mu_attr)
    return dot(F, dir)
end

const IDEAL = FP.r_a + FP.r_b


@testset "compute_attr_rep_force invariants" begin
    p = SVector(0.5, 0.5)

    # zero beyond sensing range, and at (near-)zero separation
    @test attr_rep_force(p, p + SVector(IDEAL + FP.sensing_radius + 1e-6, 0.0)) == SVector(0.0, 0.0)
    @test attr_rep_force(p, p) == SVector(0.0, 0.0)
    @test attr_rep_force(p, p + SVector(1e-9, 0.0)) == SVector(0.0, 0.0)

    # saturated repulsion inside rho*ideal
    for d in range(1e-6, FP.rho * IDEAL; length=10)
        @test signed_mag(d) ≈ FP.max_repulsion
    end

    # never more repulsive than max_repulsion; attractive outside contact
    for d in range(1e-6, IDEAL + FP.sensing_radius; length=500)
        @test signed_mag(d) >= FP.max_repulsion - 1e-12
        d > IDEAL && @test signed_mag(d) >= 0
    end

    # Newton's third law and collinearity, for random pairs incl. across the boundary
    for _ in 1:200
        a = SVector(rand(), rand())
        b = a + (IDEAL + FP.sensing_radius) * rand() * normalize(SVector(randn(), randn()))
        b = mod.(b, FP.dims)                     # may wrap across the periodic boundary
        Fab = attr_rep_force(a, b)
        Fba = attr_rep_force(b, a)
        @test Fab ≈ -Fba atol=1e-12
        v = M.shortest_vec(a, b, FP.dims)
        @test abs(v[1] * Fab[2] - v[2] * Fab[1]) < 1e-10   # 2D cross product ≈ 0
    end

    # explicit wrap case: neighbour just across the x-boundary attracts in -x direction
    a = SVector(0.005, 0.5)
    sep = IDEAL + 0.005                          # in the attractive range
    b = SVector(1.0 - (sep - a[1]), 0.5)         # reached by wrapping through x = 0
    F = attr_rep_force(a, b)
    @test F[1] < 0 && abs(F[2]) < 1e-14
end


@testset "zero OMP-OMP attraction (cf. config TEST_zero_OMP_OMP_attraction)" begin
    for d in range(IDEAL, IDEAL + FP.sensing_radius; length=200)
        @test signed_mag(d; mu_attr=0.0) == 0.0
    end
    # repulsion is unaffected by mu_attr
    @test signed_mag(0.8 * IDEAL; mu_attr=0.0) == signed_mag(0.8 * IDEAL)
end


@testset "force continuity at contact distance" begin
    ϵ = 1e-9
    # attraction branch → 0 as dist ↓ ideal
    @test abs(signed_mag(IDEAL + ϵ)) < 1e-6
    # Repulsion branch should → 0 as dist ↑ ideal. Currently fails: src/forces.jl:492 computes
    # log((d-ρI)/(1-ρ)*I) rather than log((d-ρI)/((1-ρ)*I)), giving a jump of ~mu_rep*I*log(I).
    # Switch to @test once the formula is fixed.
    @test_broken abs(signed_mag(IDEAL - ϵ)) < 1e-6
end
