# Unit tests for the force functions in src/forces.jl:
#   - compute_attr_rep_force             (pairwise attraction-repulsion)
#   - compute_inserting_substrate_force  (inserting agent <-> nascent substrate)
#
# These check qualitative invariants only (zeros, saturation, sign, bounds, continuity,
# symmetry, periodic wrap), not exact values, so they survive changes to the force law.


# =============================================================================================
# Parameters and helpers
# =============================================================================================

# Representative force parameters (cf. "force" in config/simulation/demo.json)
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

# Representative insertion parameters (cf. "insertion" in config/simulation/demo.json)
const IP = (
    mu_attr = 0.5,
    mu_rep = 0.5,
    k_C = 0.5,
)

const IDEAL = FP.r_a + FP.r_b

# force on agent at `p` due to neighbour at `q`
attr_rep_force(p, q; mu_attr=FP.mu_attr, dims=FP.dims) = M.compute_attr_rep_force(
    p, FP.r_a, q, FP.r_b, dims, FP.sensing_radius,
    mu_attr, FP.mu_rep, FP.max_repulsion, FP.rho, FP.k_C)

# force on inserting agent at `p` due to its substrate at `q` (or vice versa)
ins_force(p, q; ideal=IDEAL, dims=FP.dims) = M.compute_inserting_substrate_force(
    p, q, dims, ideal, IP.mu_attr, IP.mu_rep, FP.max_repulsion, FP.rho, IP.k_C)

# signed magnitude along the separation direction (positive = attraction)
function signed_mag(dist; mu_attr=FP.mu_attr, dir=SVector(1.0, 0.0))
    p = SVector(0.5, 0.5)
    return dot(attr_rep_force(p, p + dist * dir; mu_attr), dir)
end

function ins_signed_mag(dist; ideal=IDEAL, dir=SVector(1.0, 0.0))
    p = SVector(0.5, 0.5)
    return dot(ins_force(p, p + dist * dir; ideal), dir)
end

# Newton's third law and collinearity for random pairs up to `max_sep` apart,
# including pairs that wrap across the periodic boundary
function check_symmetry(force_fn, max_sep; n=200)
    for _ in 1:n
        a = SVector(rand(), rand())
        b = mod.(a + max_sep * rand() * normalize(SVector(randn(), randn())), FP.dims)
        Fab = force_fn(a, b)
        Fba = force_fn(b, a)
        @test Fab ≈ -Fba atol=1e-12
        v = M.shortest_vec(a, b, FP.dims)
        @test abs(v[1] * Fab[2] - v[2] * Fab[1]) < 1e-10   # 2D cross product ≈ 0
    end
end

# a neighbour at separation `sep` reached by wrapping through x = 0 attracts in -x direction
function check_wrap_attraction(force_fn, sep)
    a = SVector(0.005, 0.5)
    b = SVector(1.0 - (sep - a[1]), 0.5)
    F = force_fn(a, b)
    @test F[1] < 0 && abs(F[2]) < 1e-14
end


# =============================================================================================
# Pairwise attraction-repulsion force
# =============================================================================================

@testset "compute_attr_rep_force" begin
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

    # continuous at contact distance, from both sides
    ϵ = 1e-9
    @test abs(signed_mag(IDEAL + ϵ)) < 1e-6
    @test abs(signed_mag(IDEAL - ϵ)) < 1e-6

    # symmetry and periodic boundaries
    check_symmetry(attr_rep_force, IDEAL + FP.sensing_radius)
    check_wrap_attraction(attr_rep_force, IDEAL + 0.005)
end


@testset "zero OMP-OMP attraction (cf. config TEST_zero_OMP_OMP_attraction)" begin
    for d in range(IDEAL, IDEAL + FP.sensing_radius; length=200)
        @test signed_mag(d; mu_attr=0.0) == 0.0
    end

    # repulsion is unaffected by mu_attr
    @test signed_mag(0.8 * IDEAL; mu_attr=0.0) == signed_mag(0.8 * IDEAL)
end


# =============================================================================================
# Inserting-substrate force
# =============================================================================================

@testset "compute_inserting_substrate_force" begin
    p = SVector(0.5, 0.5)

    # zero when the ideal distance is (near-)zero, e.g. a newly created nascent agent
    for ideal in (0.0, 1e-9), d in (1e-3, 0.5 * IDEAL, IDEAL, 3 * IDEAL)
        @test ins_force(p, p + SVector(d, 0.0); ideal) == SVector(0.0, 0.0)
    end

    # zero at (near-)zero separation
    @test ins_force(p, p) == SVector(0.0, 0.0)
    @test ins_force(p, p + SVector(1e-9, 0.0)) == SVector(0.0, 0.0)

    # saturated repulsion inside rho*ideal
    for d in range(1e-6, FP.rho * IDEAL; length=10)
        @test ins_signed_mag(d) ≈ FP.max_repulsion
    end

    # never more repulsive than max_repulsion; attractive outside contact
    for d in range(1e-6, 4 * IDEAL; length=500)
        @test ins_signed_mag(d) >= FP.max_repulsion - 1e-12
        d > IDEAL && @test ins_signed_mag(d) >= 0
    end

    # no sensing-radius cutoff: still attractive well beyond where the pairwise force vanishes
    for d in (IDEAL + FP.sensing_radius + 1e-3, 2 * IDEAL, 3 * IDEAL)
        @test ins_signed_mag(d) > 0
    end

    # continuous at contact distance, from both sides
    ϵ = 1e-9
    @test abs(ins_signed_mag(IDEAL + ϵ)) < 1e-6
    @test abs(ins_signed_mag(IDEAL - ϵ)) < 1e-6

    # symmetry and periodic boundaries
    check_symmetry(ins_force, 3 * IDEAL)
    check_wrap_attraction(ins_force, 2 * IDEAL)

    # growing ideal distance at fixed separation: attraction turns into repulsion once
    # the ideal distance exceeds the separation (the substrate is pushed out as it grows)
    d = 0.05
    @test ins_signed_mag(d; ideal=0.5 * d) > 0
    @test ins_signed_mag(d; ideal=1.5 * d) < 0
end
