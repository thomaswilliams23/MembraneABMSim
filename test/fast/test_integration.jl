# Whole simulations through the public entry point (src/main.jl, src/file_io.jl): run_sim,
# write_system_state, copy_config_to_output_dir, and restarting from a checkpoint
#
# Contracts: a run writes one loadable frame per output interval plus the initial state, and a
# copy of its config; every frame is a valid system; agents and area never decrease; a seeded
# run is reproducible; a restart from a checkpoint resumes from exactly the saved state and
# carries on the output numbering.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# a small, short, busy simulation (6 frames). Polypeptides bind at once, so each BamA's first
# insertion attempt (t = 0.01) is certain to start an insertion: the agent count must grow.
tiny_run_params(; sections...) = params_with(busy_params(insertion=(PP_bind_rate=1e9,));
    init=(method="random", num_OmpA=3, num_OmpCF=2, num_LptD=2, num_BamA=2, num_LPS=30, num_PP=3,
          equilibration_time=0.01, complexes_assembled=true),
    system=(t_max=0.05, vis_dt=0.01, seed=3),
    sections...)

function run_tiny(params)
    M.run_sim(write_config(params); suppress_prints=true)
    return params.system.output_dir
end

@testset "integration" begin
    p = tiny_run_params()
    out = run_tiny(p)
    n_frames = round(Int, p.system.t_max / p.system.vis_dt) + 1

    @testset "writes every frame and a copy of the config" begin
        @test num_frames(out) == n_frames
        @test all(isfile(joinpath("out", out, "raw_data", "sys_data_$(i).jld2")) for i in 0:n_frames-1)
        @test deep_equal(M.parse_config(joinpath("out", out, "config.json")), p)
    end

    @testset "every frame is a valid system; agents and area never decrease" begin
        counts, areas = Int[], Float64[]
        for i in 0:n_frames-1
            frame = joinpath("out", out, "raw_data", "sys_data_$(i).jld2")
            state, ps = state_from_frame(p, frame)
            @test consistency_errors(state, ps) == String[]
            push!(counts, state.grid_size.num_agents)
            push!(areas, area(state))
        end
        @test issorted(counts) && issorted(areas)
        @test counts[end] > counts[1]                    # insertions started (certain, see tiny_run_params)
    end

    @testset "a seeded run is reproducible" begin
        out2 = run_tiny(params_with(p))                   # same params, new output directory
        @test all(deep_equal(load_frame(out, i), load_frame(out2, i)) for i in 0:n_frames-1)
    end

    @testset "restart from a checkpoint" begin
        last_ix = n_frames - 1
        t_checkpoint = last_ix * p.system.vis_dt
        checkpoint = joinpath("out", out, "raw_data", "sys_data_$(last_ix).jld2")
        p2 = params_with(p; init=(method="from_checkpoint", checkpoint_file=checkpoint, checkpoint_time=t_checkpoint),
                         system=(t_max=0.03,))
        out2 = run_tiny(p2)

        # resumes from exactly the saved state, and output numbering carries on from it
        @test deep_equal(load_frame(out2, last_ix), load_frame(out, last_ix))
        @test sort(readdir(joinpath("out", out2, "raw_data"))) == sort(["sys_data_$(i).jld2" for i in last_ix:last_ix+3])
        state, ps = state_from_frame(p2, joinpath("out", out2, "raw_data", "sys_data_$(last_ix + 3).jld2"))
        @test consistency_errors(state, ps) == String[]
    end
end
