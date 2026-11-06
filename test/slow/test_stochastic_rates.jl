# Rates of the stochastic biological processes (src/biology.jl), checked statistically.
#
# Each process happens with a fixed probability per step, set by its rate: 1 - exp(-rate * dt)
# (for polypeptide binding the rate is per polypeptide per unit area). So the waiting time, in
# steps, is geometric, and its sample mean must match 1 / p to within a few standard errors.
# Seeds are fixed, so the tests are deterministic; the tolerances are what keeps them meaningful
# when the implementation changes (a wrong rate is off by many standard errors).

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

const N_SAMPLES = 1500
const P_TARGET = 0.02            # per-step probability used for every process

# a sample of geometric waiting times (support 1, 2, ...) is consistent with probability p
function consistent_with_geometric(waits, p; n_sigma=4)
    n = length(waits)
    se = sqrt(1 - p) / p / sqrt(n)
    return abs(sum(waits) / n - 1 / p) <= n_sigma * se
end

# rate giving per-step probability `p`
rate_for(p, dt) = -log(1 - p) / dt

# waiting times: repeatedly reset!, then step until happened(), counting steps
function waiting_times(state, params, reset!, happened; n=N_SAMPLES)
    waits = Int[]
    t = 0.0
    for _ in 1:n
        reset!(state)
        k = 0
        while true
            t = run_steps!(state, params, 1; t0=t)
            k += 1
            happened(state) && break
        end
        push!(waits, k)
    end
    return waits
end

@testset "stochastic rates" begin
    dt = quiet_params().system.dt

    @testset "polypeptide binding" begin
        num_PP = 10^6                               # effectively constant
        p = quiet_params(insertion=(PP_bind_rate=rate_for(P_TARGET, dt) / num_PP, attempt_dt=1e3))
        state, ps = build_state(p; BamA=[(0.5, 0.5)], num_PP=num_PP, rng=Xoshiro(1))   # area 1
        BamA = only(state.agents.OMP.BamA)
        reset!(s) = (BamA.insertion_state = "free"; s.agents.num_PP = num_PP)
        waits = waiting_times(state, ps, reset!, s -> BamA.insertion_state == "stalled")
        @test consistent_with_geometric(waits, P_TARGET)
    end

    @testset "OmpA tethering" begin
        p = quiet_params(OmpA=(tether_rate=rate_for(P_TARGET, dt),))
        state, ps = build_state(p; OmpA=[(0.5, 0.5)], assembled=false, rng=Xoshiro(2))
        OmpA = only(state.agents.OMP.OmpA)
        reset!(s) = (OmpA.is_tethered = false; M.sync_grid!(s, ps))
        @test consistent_with_geometric(waiting_times(state, ps, reset!, s -> OmpA.is_tethered), P_TARGET)
    end

    @testset "BamA and LptD complex assembly" begin
        p = quiet_params(BamA=(complex_assembly_rate=rate_for(P_TARGET, dt),), LptD=(complex_assembly_rate=rate_for(P_TARGET, dt),))
        state, ps = build_state(p; BamA=[(0.3, 0.3)], LptD=[(0.7, 0.7)], assembled=false, rng=Xoshiro(3))
        BamA, LptD = only(state.agents.OMP.BamA), only(state.agents.OMP.LptD)
        reset_BamA!(s) = (BamA.is_part_of_assembled_complex = false)
        @test consistent_with_geometric(waiting_times(state, ps, reset_BamA!, s -> BamA.is_part_of_assembled_complex), P_TARGET)
        reset_LptD!(s) = (LptD.is_part_of_assembled_complex = false; LptD.is_tethered = false; M.sync_grid!(s, ps))
        @test consistent_with_geometric(waiting_times(state, ps, reset_LptD!, s -> LptD.is_part_of_assembled_complex), P_TARGET)
    end

    @testset "LPS arrival at a free LptD" begin
        # one long run: the LptD alternates between free (waiting for an arrival) and embedding
        p = quiet_params(LPS=(arrival_rate=rate_for(P_TARGET, dt), insertion_time=3dt))
        state, ps = build_state(p; LptD=[(0.5, 0.5)], rng=Xoshiro(4))
        LptD = only(state.agents.OMP.LptD)
        free_runs, run_length, t = Int[], 0, 0.0
        while length(free_runs) < 400
            t = run_steps!(state, ps, 1; t0=t)
            if LptD.insertion_state == "free"
                run_length += 1
            elseif run_length > 0
                # the promotion step ends free, then each step without an arrival: the run's
                # length is the number of steps up to and including the arrival
                push!(free_runs, run_length)
                run_length = 0
            end
        end
        @test consistent_with_geometric(free_runs[2:end], P_TARGET)    # (the first run started mid-wait)
    end

    @testset "inserted OMP type follows the insertion probabilities" begin
        probs = Dict("OmpA" => 0.5, "OmpCF" => 0.3, "BamA" => 0.2, "LptD" => 0.0)
        counts = Dict(t => 0 for t in OMP_TYPES)
        for seed in 1:8
            p = single_insertion_params("OmpA"; OmpA=(insertion_prob=0.5,), OmpCF=(insertion_prob=0.3,),
                                        BamA=(insertion_prob=0.2,), LptD=(insertion_prob=0.0,))
            BamAs = [(x, y) for x in 0.4:0.8:5.2 for y in 0.4:0.8:5.2]
            state, ps = build_state(p; BamA=BamAs, num_PP=length(BamAs), L=6.0, rng=Xoshiro(seed))
            run_steps!(state, ps, 2)
            for n in state.agents.nascent.nascent_OMP
                counts[n.OMP_type] += 1
            end
        end
        n = sum(values(counts))
        @test n == 8 * 49
        for t in OMP_TYPES
            q = probs[t]
            @test abs(counts[t] / n - q) <= 4 * sqrt(q * (1 - q) / n) + 1e-12
        end
    end
end
