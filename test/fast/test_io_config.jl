# Configuration and output files (src/file_io.jl, src/initialise.jl, src/utils.jl, src/sweep.jl):
#   parse_config, check_params, change_field, modify_params, max_steps_between_grid_sync,
#   set_up_output_directory, write_system_state
#
# Contracts: configs round-trip through JSON; bad config files are rejected; optional keys can
# be left out; check_params warns when attraction can overpower the repulsion cap; parameter
# editing changes exactly the field asked for; saved states load back unchanged.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "io and config" begin

    @testset "parse_config: demo config, JSON round trip" begin
        p = M.parse_config(DEMO_CONFIG)
        @test p isa M.AllParams

        # optional keys may be left out, and parse as nothing
        d = JSON3.read(read(DEMO_CONFIG, String), Dict{String, Any})
        for (section, key) in (("system", "max_steps_between_grid_sync"), ("system", "max_hole_radius"), ("system", "device"))
            delete!(d[section], key)
        end
        no_optional = joinpath(mktempdir(), "config.json")
        write(no_optional, JSON3.write(d))
        q = M.parse_config(no_optional)
        @test isnothing(q.system.max_steps_between_grid_sync) && isnothing(q.system.max_hole_radius) && isnothing(q.system.device)
        p2 = params_with(force=(temperature=0.5,), system=(max_steps_between_grid_sync=7,))
        @test deep_equal(M.parse_config(write_config(p2)), p2)
    end

    @testset "parse_config rejects missing and non-JSON files" begin
        @test_throws ErrorException M.parse_config(joinpath(mktempdir(), "missing.json"))
        not_json = joinpath(mktempdir(), "config.txt")
        write(not_json, read(DEMO_CONFIG, String))
        @test_throws ErrorException M.parse_config(not_json)
    end

    @testset "max_steps_between_grid_sync: default and override" begin
        @test M.max_steps_between_grid_sync(params_with()) == M.DEFAULT_MAX_STEPS_BETWEEN_GRID_SYNC
        @test M.max_steps_between_grid_sync(params_with(system=(max_steps_between_grid_sync=3,))) == 3
    end

    @testset "check_params warns when attraction can overpower repulsion" begin
        weak = params_with(force=(mu_attr_LPS_LPS=1e-4, mu_attr_OMP_LPS=1e-4, mu_attr_OMP_OMP=1e-4))
        @test_logs M.check_params(weak)                                   # weak attraction: no warnings
        for (field, pattern) in ((:mu_attr_LPS_LPS, r"LPS-LPS"), (:mu_attr_OMP_LPS, r"OMP-LPS"), (:mu_attr_OMP_OMP, r"OMP-OMP"))
            strong = params_with(force=NamedTuple{(field,)}((1e3,)))
            @test_logs (:warn, pattern) match_mode=:any M.check_params(strong)
        end
    end

    @testset "change_field / modify_params change exactly one field" begin
        p = params_with()
        q = M.change_field(p.force, "mu_rep", 0.123)
        @test q.mu_rep == 0.123
        @test all(getfield(q, f) == getfield(p.force, f) for f in fieldnames(M.ForceParams) if f != :mu_rep)
        @test_throws ErrorException M.change_field(p.force, "not_a_field", 1.0)

        p2 = M.modify_params(p, "force/mu_rep", 0.456)
        @test p2.force.mu_rep == 0.456
        @test deep_equal(M.change_field(p2, "force", p.force), p)           # nothing else changed
        @test M.modify_params(p, "system/seed", 99).system.seed == 99
    end

    @testset "set_up_output_directory" begin
        dir = mktempdir()
        M.set_up_output_directory(dir; suppress_prints=true)
        raw = joinpath("out", dir, "raw_data")
        @test isdir(raw) && isempty(readdir(raw))
        write(joinpath(raw, "old.jld2"), "x")
        M.set_up_output_directory(dir; clear_existing_output=true, suppress_prints=true)
        @test isdir(raw) && isempty(readdir(raw))
    end

    @testset "write_system_state round trip" begin
        state, ps = random_state(busy_params(); OmpA=3, LptD=1, BamA=2, LPS=20, L=1.0, num_PP=2)
        t = step_until!(s -> num_nascent(s) > 0, state, ps)       # include nascent agents
        dir = mktempdir()
        M.set_up_output_directory(dir; suppress_prints=true)
        M.write_system_state(state.agents, state.grid_size.dims, dir, 4)
        agents, dims = load_frame(dir, 4)
        @test deep_equal(agents, state.agents)
        @test dims == state.grid_size.dims
    end
end
