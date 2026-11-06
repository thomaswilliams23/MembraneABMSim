# Pairwise force laws (src/forces.jl): compute_attr_rep_force, compute_inserting_substrate_force
#
# Contracts only, never values or curve shapes, so the functional forms can change:
#   zero beyond the sensing cutoff and at coincident points; repulsive inside contact and
#   attractive outside it, with no jump at contact; repulsion capped by max_repulsion;
#   central (along the separation), equal and opposite, and invariant under translation and
#   periodic images.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

const FL = let p = quiet_params()
    (p=p, f=p.force, ins=p.insertion, dims=SVector(2.0, 2.0),   # large enough that no cutoff reaches half the domain
     radii=(p.LPS.radius, p.OmpA.radius, p.BamA.radius, p.LptD.radius))
end
const FL_RNG = Xoshiro(30)

attr_rep(p, rp, q, rq; mu_attr=FL.f.mu_attr_LPS_LPS, dims=FL.dims) = M.compute_attr_rep_force(
    p, rp, q, rq, dims, FL.f.sensing_radius, mu_attr, FL.f.mu_rep, FL.f.max_repulsion, FL.f.rho, FL.f.k_C)

spring(p, q, ideal; dims=FL.dims) = M.compute_inserting_substrate_force(
    p, q, dims, ideal, FL.ins.mu_attr, FL.ins.mu_rep, FL.f.max_repulsion, FL.f.rho, FL.ins.k_C)

# radial component of the force on an agent at the centre due to one at distance d along `dir`
# (positive = attraction towards it, negative = repulsion away from it)
function radial(force_at, d; dir=SVector(1.0, 0.0))
    c = FL.dims ./ 2
    return dot(force_at(c, mod.(c + d * dir, FL.dims)), dir)
end

# no jump at contact: the force tends to zero from both sides
passes_through_zero(g, d0) = all(abs(g(d0 * (1 + s * e))) < 1e-3 * abs(FL.f.max_repulsion) for s in (-1, 1) for e in (1e-6, 1e-8))

# the force laws must be central, equal and opposite, translation invariant and periodic
function check_pair_symmetries(force, max_sep; n=300)
    for _ in 1:n
        p = FL.dims .* SVector(rand(FL_RNG), rand(FL_RNG))
        q = mod.(p + max_sep * rand(FL_RNG) * normalize(SVector(randn(FL_RNG), randn(FL_RNG))), FL.dims)
        F = force(p, q)
        v = M.shortest_vec(p, q, FL.dims)
        @test abs(v[1] * F[2] - v[2] * F[1]) <= 1e-12 * (1 + norm(F)) * norm(v)   # along the separation
        @test force(q, p) ≈ -F atol=1e-12                                          # equal and opposite
        shift = FL.dims .* SVector(rand(FL_RNG), rand(FL_RNG))
        @test force(mod.(p + shift, FL.dims), mod.(q + shift, FL.dims)) ≈ F atol=1e-10
        @test force(p, q + SVector(2.0, -4.0)) ≈ F atol=1e-10                     # periodic image
    end
end

@testset "force laws" begin

    @testset "compute_attr_rep_force" begin
        for ra in FL.radii, rb in FL.radii, mu_attr in (FL.f.mu_attr_LPS_LPS, FL.f.mu_attr_OMP_OMP)
            contact = ra + rb
            cutoff = contact + FL.f.sensing_radius
            g(d) = radial((p, q) -> attr_rep(p, ra, q, rb; mu_attr), d)

            # zero at coincident points and beyond the cutoff
            @test attr_rep(SVector(0.4, 0.4), ra, SVector(0.4, 0.4), rb) == SVector(0.0, 0.0)
            @test g(cutoff * 1.001) == 0 && g(min(cutoff * 1.5, 0.99)) == 0

            # repulsive inside contact, attractive between contact and the cutoff
            inside = range(1e-6, contact; length=400)[1:end-1]
            outside = range(contact, cutoff; length=400)[2:end-1]
            @test all(g(d) < 0 for d in inside)
            @test all(g(d) > 0 for d in outside)
            @test abs(g(contact)) < 1e-12

            # repulsion never exceeds the cap
            @test all(g(d) >= FL.f.max_repulsion - 1e-12 for d in inside)

            # repulsion turns into attraction without a jump
            @test passes_through_zero(g, contact)
        end

        # no attraction at all when mu_attr is zero
        r = FL.p.LPS.radius
        @test all(radial((p, q) -> attr_rep(p, r, q, r; mu_attr=0.0), d) <= 0 for d in range(1e-6, 2r + FL.f.sensing_radius; length=500))

        # equal and opposite needs the radii swapped with the positions
        ra, rb = FL.p.BamA.radius, FL.p.LPS.radius
        check_pair_symmetries((p, q) -> attr_rep(p, ra, q, rb), ra + rb + FL.f.sensing_radius)
        @test all(attr_rep(q, rb, p, ra) ≈ -attr_rep(p, ra, q, rb)
                  for (p, q) in ((SVector(0.1, 0.1), SVector(0.25, 0.2)), (SVector(0.02, 0.5), SVector(1.95, 0.5))))
    end

    @testset "compute_inserting_substrate_force" begin
        sens = FL.f.sensing_radius
        for ideal in (0.02, 0.1, 0.2, 0.3)
            g(d) = radial((p, q) -> spring(p, q, ideal), d)
            reach = ideal + 2sens            # well past where attraction-repulsion would cut off

            # zero at coincident points; repulsive inside the ideal distance, attractive outside
            @test spring(SVector(0.4, 0.4), SVector(0.4, 0.4), ideal) == SVector(0.0, 0.0)
            @test all(g(d) < 0 for d in range(1e-6, ideal; length=300)[1:end-1])
            @test all(g(d) > 0 for d in range(ideal, reach; length=300)[2:end])
            @test abs(g(ideal)) < 1e-12

            # capped repulsion, no jump at the ideal distance, and no sensing cutoff: the partner
            # is still felt beyond the range where attraction-repulsion stops
            @test all(g(d) >= FL.f.max_repulsion - 1e-12 for d in range(1e-6, ideal; length=300))
            @test passes_through_zero(g, ideal)
            @test g(ideal + 1.5sens) != 0

            check_pair_symmetries((p, q) -> spring(p, q, ideal), reach)
        end
    end
end
