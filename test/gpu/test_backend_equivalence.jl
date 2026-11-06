# GPU backends (src/metal/, src/cuda/, src/sim_state.jl) against the CPU and the brute-force
# oracle. Every testset runs once per backend whose device is present; a missing device is
# reported as skipped.
#
# Contracts: data copied to the device and back is unchanged; force steps on the device give the
# same motion as the CPU (to Float32 precision), including on a stale grid; with insertions
# the device run makes the same biological decisions as the CPU run and stays consistent.
#
# Only the backend-dispatched operations (step!, advance_positions!, copy_to_host!,
# copy_to_device!, sync_grid!) are used, so the same code tests Metal and CUDA.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

const GPU_BACKENDS = ("metal", "cuda")

"is the device for `backend` present and usable?"
function gpu_available(backend)
    try
        return backend == "metal" ? M.Metal.functional() : M.CUDA.functional()
    catch
        return false
    end
end

# the same random membrane built on the CPU and on `backend`
twin_states(params, backend; n...) = (
    random_state(params; n..., L=1.0, rng=Xoshiro(11), device="cpu"),
    random_state(params; n..., L=1.0, rng=Xoshiro(11), device=backend))

# largest distance between the same agent in two states (optionally only agents passing `which`)
function max_position_gap(a::M.SimState, b::M.SimState; which=i -> true)
    pa, pb = positions_by_index(a), positions_by_index(b)
    keys(pa) == keys(pb) || return Inf
    ixs = filter(which, collect(keys(pa)))
    return isempty(ixs) ? 0.0 : maximum(M.shortest_distance(pa[i], pb[i], a.grid_size.dims) for i in ixs)
end

# Float32 resolution of a position in a domain of size ~1
const F32_POS_TOL = 10 * eps(Float32)

@testset "gpu backends" begin
    for backend in GPU_BACKENDS
        @testset "$backend" begin
            if !gpu_available(backend)
                @info "$backend device not available: skipping its GPU tests"
                @test_skip "$backend device available"
                continue
            end

            @testset "a freshly initialised state has no pending rebuild" begin
                _, (gpu, _) = twin_states(quiet_params(), backend; OmpA=2, LPS=20)
                @test !gpu.grid_size.num_agents_changed
            end

            @testset "copies to the device and back are unchanged" begin
                (cpu, ps), (gpu, _) = twin_states(quiet_params(), backend; OmpA=3, OmpCF=2, LptD=1, BamA=2, LPS=60)
                @test gpu.backend isa M.AbstractSimBackend && !(gpu.backend isa M.SimBackendCPU)
                for _ in 1:2
                    M.copy_to_device!(gpu)
                    M.copy_to_host!(gpu)
                    @test max_position_gap(cpu, gpu) <= F32_POS_TOL
                end
                @test consistency_errors(gpu, ps) == String[]
            end

            @testset "force steps agree with the brute-force oracle, including on a stale grid" begin
                p = quiet_params(system=(max_steps_between_grid_sync=10^6,))
                _, (gpu, ps) = twin_states(p, backend; OmpA=4, OmpCF=3, LptD=1, BamA=2, LPS=150)
                for n_before in (0, 300)                     # fresh grid, then stale grid
                    t = run_steps!(gpu, ps, n_before)
                    M.copy_to_host!(gpu)
                    before = positions_by_index(gpu)
                    expected = expected_displacements(gpu, ps)
                    M.advance_positions!(gpu, ps, t + ps.system.dt)
                    M.copy_to_host!(gpu)
                    observed = observed_displacements(before, gpu)
                    worst = maximum(norm(observed[i] - expected[i]) for i in keys(expected))
                    @test worst <= 1e-3 * maximum(norm, values(expected)) + F32_POS_TOL
                end
            end

            @testset "trajectories agree with the CPU" begin
                # Float32 rounding differences grow chaotically from an overlapping start, so
                # compare over a short horizon (each step is checked against the oracle above)
                (cpu, ps), (gpu, _) = twin_states(quiet_params(), backend; OmpA=4, OmpCF=2, LptD=1, BamA=2, LPS=100)
                run_steps!(cpu, ps, 20)
                run_steps!(gpu, ps, 20)
                M.copy_to_host!(gpu)
                @test max_position_gap(cpu, gpu) < 1e-5           # ~8x Float32 resolution (measured ~2e-7)
                @test cpu.grid_size.dims == gpu.grid_size.dims
            end

            @testset "with insertions: same decisions as the CPU, consistent" begin
                # Positions are not compared here: over a long busy run Float32 rounding differences
                # grow chaotically. Decisions do not depend on positions with guaranteed insertion.
                (cpu, ps), (gpu, _) = twin_states(busy_params(), backend; OmpA=3, OmpCF=2, LptD=2, BamA=3, LPS=60, num_PP=4)
                t = 0.0
                for _ in 1:15
                    t_next = run_steps!(cpu, ps, 100; t0=t)
                    run_steps!(gpu, ps, 100; t0=t)
                    t = t_next
                    M.copy_to_host!(gpu)
                    @test agent_counts(gpu) == agent_counts(cpu)
                    @test insertion_states_of(gpu) == insertion_states_of(cpu)
                    @test consistency_errors(gpu, ps) == String[]
                end
                @test agent_counts(cpu).OmpA + agent_counts(cpu).LPS > 3 + 60     # it was busy (effectively certain, see busy_params)
                @test isapprox(area(gpu), area(cpu); rtol=1e-5)
            end

            @testset "a new nascent LPS starts at its initial distance from its LptD" begin
                # New nascent agents are placed relative to the inserter's position in the host
                # structs, which on a GPU can be stale. Here overlapping LPS push the LptD around
                # with every host copy suppressed (no output, no insertion attempts, no rebuilds),
                # then an LPS arrives: it must still start at its initial distance from the LptD.
                p0 = quiet_params(system=(vis_dt=1e3, max_steps_between_grid_sync=10^6), insertion=(attempt_dt=1e3,))
                (cpu, ps), (gpu, _) = twin_states(p0, backend; LptD=1, LPS=250)
                start = only(cpu.agents.OMP.LptD).position
                t = run_steps!(cpu, ps, 1000)
                run_steps!(gpu, ps, 1000)
                drift = M.shortest_distance(start, only(cpu.agents.OMP.LptD).position, cpu.grid_size.dims)
                @test drift > 0.02                                # the scenario really moves the LptD

                arrive = params_with(ps; LPS=(arrival_rate=1e9,))
                ideal_start = ps.LptD.radius - ps.LPS.radius
                offset_from_start(s) = (n = only(s.agents.nascent.nascent_LPS);
                    abs(M.shortest_distance(n.position, only(s.agents.OMP.LptD).position, s.grid_size.dims) - ideal_start))
                run_steps!(cpu, arrive, 1; t0=t)
                run_steps!(gpu, arrive, 1; t0=t)
                M.copy_to_host!(gpu)
                tol = 0.1drift      # one step's growth and motion is far smaller (measured: CPU ~3e-6, GPU ~1e-6)
                @test offset_from_start(cpu) < tol
                @test offset_from_start(gpu) < tol
            end
        end
    end
end
