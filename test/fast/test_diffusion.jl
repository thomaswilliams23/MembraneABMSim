# Diffusion modes (src/forces.jl): compute_diffusion!
#
# Contracts: nothing diffuses at zero temperature or with diffusion off; in "LPS_only" mode LPS
# (including nascent LPS) diffuse and OMPs (including nascent OMPs) stay put; modes that would
# diffuse OMPs are rejected rather than silently run (OMP diffusion is not supported); diffusion
# keeps the system consistent. (Diffusion statistics are checked in slow/test_physics.jl.)

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

# a quiet membrane holding a nascent OMP and a nascent LPS, part-way through their insertions
function state_with_nascent(force_overrides)
    p = busy_params(OmpA=(tether_rate=0.0,), LPS=(insertion_time=10.0,), insertion=(OMP_assembly_rate=0.01, PP_bind_rate=1e9))
    state, ps = random_state(p; OmpA=3, OmpCF=2, LptD=1, BamA=1, LPS=40, L=1.0, num_PP=1, rng=Xoshiro(2))
    step_until!(s -> !isempty(s.agents.nascent.nascent_OMP) && !isempty(s.agents.nascent.nascent_LPS), state, ps)
    return state, params_with(ps; force=force_overrides)
end

# agents that moved during one diffusion step, split into LPS and OMPs
function diffuse_once!(state, ps)
    before = positions_by_index(state)
    M.compute_diffusion!(state.agents, state.flat, state.grid_size, ps; rng=Xoshiro(3))
    moved = Set(i for (i, d) in observed_displacements(before, state) if norm(d) > 0)
    lps = [a.index for a in all_agents(state) if !is_OMP(a)]
    omps = [a.index for a in all_agents(state) if is_OMP(a)]
    return (lps_moved=count(in(moved), lps), n_lps=length(lps), omps_moved=count(in(moved), omps))
end

@testset "diffusion" begin

    @testset "nothing moves at zero temperature or with diffusion off" begin
        for force in ((temperature=0.0, diffusion_mode="LPS_only"), (temperature=1e-3, diffusion_mode="off"))
            state, ps = state_with_nascent(force)
            r = diffuse_once!(state, ps)
            @test r.lps_moved == 0 && r.omps_moved == 0
        end
    end

    @testset "LPS_only: every LPS diffuses, no OMP does" begin
        state, ps = state_with_nascent((temperature=1e-3, diffusion_mode="LPS_only"))
        r = diffuse_once!(state, ps)
        @test r.lps_moved == r.n_lps                      # including the nascent LPS
        @test r.omps_moved == 0                           # including the nascent OMP
        @test consistency_errors(state, ps) == String[]   # flat data and grid margins kept up to date
    end

    @testset "modes that would diffuse OMPs are rejected" begin
        for mode in ("all", nothing)
            state, ps = state_with_nascent((temperature=1e-3, diffusion_mode=mode))
            @test_throws ErrorException M.compute_diffusion!(state.agents, state.flat, state.grid_size, ps; rng=Xoshiro(3))
        end
    end
end
