# Parameter sweeps (src/sweep.jl, src/main.jl, src/file_io.jl): build_sim_dict,
# make_config_this_sim, parse_sweep_config, run_sweep
#
# Contracts: a sweep runs every combination of the swept values for every replicate, each with
# its own seed and output directory, and each simulation's saved config carries its overrides.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# writes a sweep config over `sweep` (path => (short_name, values)) and returns its path
function write_sweep_config(sweep, num_reps, default_config, output_base_dir)
    d = Dict("sweep_params" => Dict(k => Dict("short_name" => s, "values" => v) for (k, (s, v)) in sweep),
             "num_reps" => num_reps, "default_config" => default_config, "output_base_dir" => output_base_dir)
    path = joinpath(mktempdir(), "sweep.json")
    write(path, JSON3.write(d))
    return path
end

@testset "sweep" begin

    @testset "build_sim_dict: every combination, every replicate, unique seeds" begin
        sweep = Dict("force/mu_attr_LPS_LPS" => ("muLL", [0.1, 0.5]), "insertion/attempt_dt" => ("adt", [0.1, 0.2, 0.3]))
        cfg = M.parse_sweep_config(write_sweep_config(sweep, 2, DEMO_CONFIG, mktempdir()))
        sims = M.build_sim_dict(cfg)

        @test length(sims) == 2 * 3 * 2
        combos = [(c["force/mu_attr_LPS_LPS"], c["insertion/attempt_dt"]) for c in values(sims)]
        @test sort(unique(combos)) == sort([(a, b) for a in [0.1, 0.5] for b in [0.1, 0.2, 0.3]])
        @test all(count(==(c), combos) == 2 for c in unique(combos))      # one per replicate
        @test allunique(c["system/seed"] for c in values(sims))

        # each simulation's config applies its overrides and its own output directory
        base = M.parse_config(DEMO_CONFIG)
        for (path, changes) in sims
            q = M.make_config_this_sim(base, changes, joinpath("sweep_out", path))
            @test q.force.mu_attr_LPS_LPS == changes["force/mu_attr_LPS_LPS"]
            @test q.insertion.attempt_dt == changes["insertion/attempt_dt"]
            @test q.system.seed == changes["system/seed"]
            @test q.system.output_dir == joinpath("sweep_out", path)
        end
    end

    @testset "run_sweep runs and saves each simulation" begin
        default = params_with(quiet_params();
            init=(method="random", num_OmpA=2, num_OmpCF=1, num_LptD=1, num_BamA=1, num_LPS=10, equilibration_time=0.0),
            system=(t_max=0.002, vis_dt=0.001, seed=1))
        base_dir = mktempdir()
        sweep = Dict("force/mu_rep" => ("murep", [0.04, 0.06]))
        cfg_path = write_sweep_config(sweep, 2, write_config(default), base_dir)
        M.run_sweep(cfg_path; run_serially_yn=true)

        sims = M.build_sim_dict(M.parse_sweep_config(cfg_path))
        @test length(sims) == 4
        for (path, changes) in sims
            sim_out = joinpath(base_dir, path)
            @test num_frames(sim_out) == 3
            saved = M.parse_config(joinpath("out", sim_out, "config.json"))
            @test saved.force.mu_rep == changes["force/mu_rep"]
            @test saved.system.seed == changes["system/seed"]
        end
    end
end
