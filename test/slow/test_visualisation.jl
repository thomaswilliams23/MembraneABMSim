# Visualisation (src/postprocessing/vis_membrane.jl): make_snapshot, make_membrane_movie
#
# Smoke tests only: each runs on a short simulation's output and writes its file. Images are not
# compared. (In the slow tier because loading the plotting stack takes a while.)

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "visualisation" begin
    p = params_with(busy_params();
        init=(method="random", num_OmpA=3, num_OmpCF=1, num_LptD=1, num_BamA=2, num_LPS=25, num_PP=2,
              equilibration_time=0.0, complexes_assembled=true),
        system=(t_max=0.02, vis_dt=0.01, seed=4))
    M.run_sim(write_config(p); suppress_prints=true)
    out = p.system.output_dir

    M.make_snapshot(out, 0.01)
    @test isfile(joinpath("out", out, "snapshots", "snapshot_1.png"))
    @test isfile(joinpath("out", out, "vector_snapshots", "snapshot_1_vector.svg"))

    M.make_membrane_movie(out; output_extension=".gif", fps=2)
    @test filesize(joinpath("out", out, "sim.gif")) > 0
end
