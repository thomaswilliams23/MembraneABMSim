# Step-counted timing (src/utils.jl): is_interval_step, copy_data_to_cpu_yn
#
# Contracts: an interval event fires once per interval, on whole steps, for awkward dt values
# where floating-point time comparisons would drift; GPU data is copied back to the host
# exactly when the next step needs it.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "timing" begin

    @testset "is_interval_step fires once per interval" begin
        for (dt, interval, T) in ((1e-4, 0.1, 30.0), (0.003, 0.03, 3.0), (0.1, 0.1, 5.0), (1e-3, 0.0137, 2.0))
            n_steps = round(Int, T / dt)
            steps_per_interval = max(1, round(Int, interval / dt))
            fired = [ix for ix in 1:n_steps if M.is_interval_step(ix * dt, interval, dt)]

            @test length(fired) == n_steps ÷ steps_per_interval
            @test all(diff(fired) .== steps_per_interval)                 # evenly spaced
            @test isempty(fired) || first(fired) == steps_per_interval    # first at one interval
        end
        # an interval shorter than a step fires every step
        @test all(M.is_interval_step(ix * 0.01, 0.001, 0.01) for ix in 1:100)
        # time zero is on every interval
        @test M.is_interval_step(0.0, 0.1, 1e-4)
    end

    @testset "copy_data_to_cpu_yn: copy back exactly when the next step needs host data" begin
        p = params_with(system=(dt=0.01, vis_dt=0.1), insertion=(attempt_dt=0.05,))
        dt = p.system.dt
        state, p = build_state(p; BamA=[(0.3, 0.3), (0.7, 0.7)])
        BamAs = state.agents.OMP.BamA

        for ix in 0:200
            t = ix * dt
            output_next = M.is_interval_step(t + dt, p.system.vis_dt, dt)
            attempt_next = M.is_interval_step(t + dt, p.insertion.attempt_dt, dt)

            # must copy before output, and before a stalled BamA's insertion attempt
            BamAs[2].insertion_state = "stalled"
            @test M.copy_data_to_cpu_yn(BamAs, p, t) >= (output_next || attempt_next)

            # with nothing needing host data, don't copy (copying every step defeats the GPU)
            foreach(b -> b.insertion_state = "free", BamAs)
            output_next || @test !M.copy_data_to_cpu_yn(BamAs, p, t)
        end
    end
end
