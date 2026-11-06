# Postprocessing metrics (src/postprocessing/metrics.jl, src/postprocessing/postprocessing_utils.jl)
#
# Each metric is evaluated on a hand-built membrane whose answer is known, including
# arrangements that wrap around the periodic boundary. Trajectory metrics and the analysis
# drivers are run on a short simulation and checked for shape and basic sanity.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

metric(f, state, params) = f(state.agents, state.grid_size.dims, params)

@testset "metrics" begin
    p = quiet_params()
    rL, rA, rB = p.LPS.radius, p.OmpA.radius, p.BamA.radius
    sens = p.force.sensing_radius

    @testset "counts and sizes" begin
        state, ps = build_state(p; OmpA=[(0.2, 0.2), (0.4, 0.2)], OmpCF=[(0.7, 0.7)], BamA=[(0.2, 0.7)],
                                LptD=[(0.7, 0.2)], LPS=[(0.5, 0.5), (0.5, 0.9), (0.9, 0.5)], L=1.2)
        @test metric(M.get_membrane_size, state, ps) ≈ 1.44
        @test metric(M.get_num_agents, state, ps) == Dict("OmpA" => 2, "OmpCF" => 1, "LptD" => 1, "BamA" => 1, "LPS" => 3)
        @test metric(M.get_num_OmpA_agents, state, ps) == 2
        @test metric(M.get_num_LPS_agents, state, ps) == 3
        ideal = (2disk_area(rA) + disk_area(ps.OmpCF.radius) + disk_area(rB) + disk_area(ps.LptD.radius) + 3disk_area(rL)) / 1.44
        @test metric(M.get_ideal_domain_coverage, state, ps) ≈ ideal
        @test metric(M.get_BAM_states, state, ps) == ["free"]
    end

    @testset "LPS bordering OMPs, including across the boundary" begin
        touching = rL + rA + 0.5rL                        # within the bordering distance
        state, ps = build_state(p; OmpA=[(0.02, 0.5)],
                                LPS=[(mod(0.02 - touching, 1.0), 0.5), (0.02 + touching, 0.5), (0.5, 0.5), (0.6, 0.9)])
        @test metric(M.get_num_LPS_bordering_OMP, state, ps) == 2
        @test metric(M.get_prop_LPS_bordering_OMP, state, ps) ≈ 0.5
    end

    @testset "LPS clusters: touching chains, joined across the boundary" begin
        chain(x0, y, n) = [(mod(x0 + 2rL * i, 1.0), y) for i in 0:n-1]
        state, ps = build_state(p; LPS=vcat(chain(0.95, 0.2, 4), chain(0.3, 0.6, 3), [(0.7, 0.85)]))
        sizes = sort(length.(metric(M.get_LPS_clusters, state, ps)))
        @test sizes == [1, 3, 4]
        @test sort(vcat(metric(M.get_LPS_clusters, state, ps)...)) == 1:8     # every LPS in exactly one cluster
    end

    @testset "domain coverage approximates the covered area" begin
        state, ps = build_state(p; OmpCF=[(0.3, 0.3)], LPS=[(0.7, 0.7)])
        exact = disk_area(ps.OmpCF.radius) + disk_area(rL)
        @test metric(M.get_domain_coverage, state, ps) ≈ exact rtol=0.1
        @test metric(M.get_domain_coverage, state, ps) ≈ metric(M.get_ideal_domain_coverage, state, ps) rtol=0.1
        # overlapping agents cover less than their total area
        state, ps = build_state(p; OmpCF=[(0.3, 0.3), (0.32, 0.3)])
        @test metric(M.get_domain_coverage, state, ps) < 0.8 * metric(M.get_ideal_domain_coverage, state, ps)
    end

    @testset "BamA: gaps between them (across the boundary) and access to LPS" begin
        state, ps = build_state(p; BamA=[(0.2, 1.0), (1.4, 1.0)], LPS=[(0.2 + rB + rL + 0.5sens, 1.0)], L=2.0)
        gaps = metric(M.get_distance_between_BAMs, state, ps)
        @test gaps[1, 2] ≈ 0.8 - 2rB              # 0.8 apart the short way round
        @test gaps[1, 2] == gaps[2, 1]
        @test metric(M.get_BAM_stalled_status, state, ps) == [false, true]   # only the first can reach an LPS
        counts = metric(M.get_num_new_and_old_LPS_in_sensing_radius, state, ps)
        @test counts[1]["old"] == 1 && counts[1]["new"] == 0 && counts[2]["old"] == 0

        # an LPS that arrived during the run counts as new; old/new LPS distances
        state.agents.LPS[1].arrival_time = 0.5
        counts = metric(M.get_num_new_and_old_LPS_in_sensing_radius, state, ps)
        @test counts[1]["new"] == 1 && counts[1]["old"] == 0
    end

    @testset "LPS relative to one BamA and one LptD" begin
        state, ps = build_state(p; BamA=[(0.2, 0.5)], LptD=[(0.8, 0.5)], LPS=[(0.7, 0.2), (0.75, 0.8), (0.3, 0.5), (0.45, 0.5)])
        @test metric(M.get_prop_LPS_closer_to_Lpt_than_BAM, state, ps) ≈ 0.5
        @test metric(M.get_prop_old_LPS_closer_to_Lpt_than_BAM, state, ps) ≈ 0.5
        dists_Lpt = [M.shortest_distance(l.position, SVector(0.8, 0.5), state.grid_size.dims) for l in state.agents.LPS]
        @test metric(M.get_mean_old_LPS_distance_to_Lpt, state, ps) ≈ sum(dists_Lpt) / 4
        dists_BAM = [M.shortest_distance(l.position, SVector(0.2, 0.5), state.grid_size.dims) for l in state.agents.LPS]
        @test metric(M.get_mean_old_LPS_distance_to_BAM, state, ps) ≈ sum(dists_BAM) / 4
    end

    @testset "trajectory metrics and analysis drivers on a short run" begin
        rp = params_with(busy_params();
            init=(method="random", num_OmpA=2, num_OmpCF=1, num_LptD=1, num_BamA=2, num_LPS=20, num_PP=3, equilibration_time=0.0),
            system=(t_max=0.03, vis_dt=0.01, seed=2))
        cfg = write_config(rp)
        M.run_sim(cfg; suppress_prints=true)
        n_frames = 4
        out_path = joinpath("out", rp.system.output_dir)

        series = M.get_time_series(M.get_num_LPS_agents, out_path, rp)
        @test length(series) == n_frames && issorted(series)

        sq = M.get_squared_displacement(rp)
        @test all(all(v[:, 1] .== 0) && all(v .>= 0) for v in values(sq))        # zero at the start
        by_BAM = M.get_num_OmpA_agents_by_BAM(rp)
        @test size(by_BAM, 1) == n_frames && all(diff(by_BAM; dims=1) .>= 0)     # cumulative counts
        @test length(M.get_cutoff_time(rp)) == 2
        @test length(M.get_LptD_insertion_times(rp)) == length(load_frame(rp.system.output_dir, n_frames - 1)[1].OMP.LptD)

        M.analyse_sim(M.get_num_agents, "num_agents", cfg; evaluate_final_state_yn=true)
        saved = JLD2.load(joinpath(out_path, "processed_data", "num_agents.jld2"), "result")
        @test saved == M.get_num_agents(load_frame(rp.system.output_dir, n_frames - 1)..., rp)
    end
end
