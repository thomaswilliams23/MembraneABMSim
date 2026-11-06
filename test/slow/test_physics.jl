# Physical sanity of the dynamics, with wide tolerances so the force laws and diffusion model
# can change without breaking these tests.
#
# Contracts: diffusion is unbiased, isotropic and linear in time, with mean squared displacement
# proportional to temperature; a pair of agents settles at contact (where the force vanishes)
# whether started apart or overlapping; a random packing equilibrates out of the hard core with
# no attraction, and without deep overlaps with attraction weak enough that check_params is silent.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# squared displacement of every agent since `before`
sq_displacements(before, state) = [norm(d)^2 for d in values(observed_displacements(before, state))]

# LPS on a square lattice with spacing well beyond the interaction cutoff
function sparse_lps(spacing, L)
    xs = (spacing / 2):spacing:(L - spacing / 2)
    return [(x, y) for x in xs for y in xs]
end

@testset "physics" begin

    @testset "diffusion: linear in time, proportional to temperature, isotropic, unbiased" begin
        L = 6.0
        lps = sparse_lps(0.3, L)
        msd = Dict{Tuple{Float64, Int}, Float64}()
        drift = SVector(0.0, 0.0)
        xy = [0.0, 0.0]
        for T in (1e-4, 2e-4)
            p = quiet_params(force=(temperature=T, diffusion_mode="LPS_only"))
            state, ps = build_state(p; LPS=lps, L, rng=Xoshiro(round(Int, T * 1e5)))
            before = positions_by_index(state)
            t = run_steps!(state, ps, 500)
            msd[(T, 500)] = sum(sq_displacements(before, state)) / length(lps)
            run_steps!(state, ps, 500; t0=t)
            disps = collect(values(observed_displacements(before, state)))
            msd[(T, 1000)] = sum(norm.(disps) .^ 2) / length(lps)
            if T == 1e-4
                drift = sum(disps) / length(disps)
                xy = [sum(d[1]^2 for d in disps), sum(d[2]^2 for d in disps)]
            end
        end
        n = length(lps)
        rel_se = 1.0 / sqrt(n)                          # relative standard error of a 2D MSD estimate
        @test msd[(1e-4, 1000)] / msd[(1e-4, 500)] ≈ 2 rtol=6rel_se       # linear in time
        @test msd[(2e-4, 1000)] / msd[(1e-4, 1000)] ≈ 2 rtol=6rel_se      # proportional to temperature
        @test xy[1] / xy[2] ≈ 1 rtol=6 * sqrt(2 / n)                       # isotropic
        @test norm(drift) <= 5 * sqrt(msd[(1e-4, 1000)] / n)              # unbiased
    end

    @testset "a pair settles at contact, from apart or overlapping" begin
        p = quiet_params(system=(dt=1e-3,))
        for (type_a, type_b) in (("LPS", "LPS"), ("OmpCF", "LPS"), ("OmpCF", "OmpCF"))
            r_a, r_b = type_radius(p, type_a), type_radius(p, type_b)
            contact = r_a + r_b
            for start in (0.85contact, contact + 0.5p.force.sensing_radius)
                kw = type_a == type_b ? (; Symbol(type_a) => [(0.5, 0.5), (0.5 + start, 0.5)]) :
                                        (; Symbol(type_a) => [(0.5, 0.5)], Symbol(type_b) => [(0.5 + start, 0.5)])
                state, ps = build_state(p; kw..., L=1.5)
                run_steps!(state, ps, 20_000)
                a, b = all_agents(state)
                @test M.shortest_distance(a.position, b.position, state.grid_size.dims) ≈ contact rtol=0.01
            end
        end
    end

    @testset "random packing equilibrates without deep overlaps" begin
        # closest pair, as a fraction of its contact distance
        function closest_pair(state, ps)
            as = all_agents(state)
            return minimum(M.shortest_distance(a.position, b.position, state.grid_size.dims) /
                           (current_radius(ps, a) + current_radius(ps, b))
                           for (i, a) in enumerate(as) for b in as[i+1:end])
        end
        settle(p) = (sp = random_state(p; OmpA=4, OmpCF=3, BamA=2, LPS=200, L=1.2, rng=Xoshiro(5));
                     run_steps!(sp[1], sp[2], 10_000); sp)

        # without attraction, every pair is pushed out of the hard core
        state, ps = settle(quiet_params(force=(mu_attr_LPS_LPS=0.0, mu_attr_OMP_LPS=0.0, mu_attr_OMP_OMP=0.0)))
        @test closest_pair(state, ps) >= ps.force.rho
        @test consistency_errors(state, ps) == String[]

        # with attraction in the range check_params accepts, neighbours can squeeze a pair a little
        # way into the hard core (the repulsion there is capped), but never deeply
        p = quiet_params(force=(mu_attr_LPS_LPS=0.3, mu_attr_OMP_LPS=0.025, mu_attr_OMP_OMP=0.025))
        @test_logs M.check_params(p)
        state, ps = settle(p)
        @test closest_pair(state, ps) >= 0.95 * ps.force.rho
        @test consistency_errors(state, ps) == String[]
    end
end
