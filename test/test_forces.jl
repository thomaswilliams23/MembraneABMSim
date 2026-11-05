# Unit tests for the functions in src/forces.jl, in source order:
#   resolve_forces_cpu!, compile_flat_system_data_cpu!, compute_next_position,
#   tally_attr_rep_forces, compute_attr_rep_force, compute_inserting_substrate_force,
#   compute_diffusion!
#
# These check invariants only (zeros, saturation, sign, bounds, continuity, symmetry, periodic
# wrap, "aggregate = sum of pairwise calls"), never exact force values, so they survive
# changes to the force laws.

include(joinpath(@__DIR__, "setup.jl"))


# representative force parameters (cf. "force" in config/simulation/demo.json)
const FP = (
    dims = SVector(1.0, 1.0),
    r_a = 0.02,
    r_b = 0.015,
    sensing_radius = 0.02,
    mu_attr = 0.1,
    mu_rep = 1.0,
    max_repulsion = -0.2,
    rho = 0.5,
    k_C = 3.0,
)

# representative insertion parameters (cf. "insertion" in config/simulation/demo.json)
const IP = (
    mu_attr = 0.5,
    mu_rep = 0.5,
    k_C = 0.5,
)

const IDEAL = FP.r_a + FP.r_b


# force on agent at `p` due to neighbour at `q`
attr_rep_force(p, q; mu_attr=FP.mu_attr, dims=FP.dims) = M.compute_attr_rep_force(
    p, FP.r_a, q, FP.r_b, dims, FP.sensing_radius,
    mu_attr, FP.mu_rep, FP.max_repulsion, FP.rho, FP.k_C)

# force on inserting agent at `p` due to its substrate at `q` (or vice versa)
ins_force(p, q; ideal=IDEAL, dims=FP.dims) = M.compute_inserting_substrate_force(
    p, q, dims, ideal, IP.mu_attr, IP.mu_rep, FP.max_repulsion, FP.rho, IP.k_C)

# signed magnitude along the separation direction (positive = attraction)
function signed_mag(dist; mu_attr=FP.mu_attr, dir=SVector(1.0, 0.0))
    p = SVector(0.5, 0.5)
    return dot(attr_rep_force(p, p + dist * dir; mu_attr), dir)
end

function ins_signed_mag(dist; ideal=IDEAL, dir=SVector(1.0, 0.0))
    p = SVector(0.5, 0.5)
    return dot(ins_force(p, p + dist * dir; ideal), dir)
end

# Newton's third law and collinearity for random pairs up to `max_sep` apart,
# including pairs that wrap across the periodic boundary
function check_symmetry(force_fn, max_sep; n=200)
    for _ in 1:n
        a = SVector(rand(), rand())
        b = mod.(a + max_sep * rand() * normalize(SVector(randn(), randn())), FP.dims)
        Fab = force_fn(a, b)
        Fba = force_fn(b, a)
        @test Fab ≈ -Fba atol=1e-12
        v = M.shortest_vec(a, b, FP.dims)
        @test abs(v[1] * Fab[2] - v[2] * Fab[1]) < 1e-10   # 2D cross product ≈ 0
    end
end

# a neighbour at separation `sep` reached by wrapping through x = 0 attracts in -x direction
function check_wrap_attraction(force_fn, sep)
    a = SVector(0.005, 0.5)
    b = SVector(1.0 - (sep - a[1]), 0.5)
    F = force_fn(a, b)
    @test F[1] < 0 && abs(F[2]) < 1e-14
end


# --- whole-system helpers (agents, grid and flat data, as built by the simulation) ---

# build agents of each type at the given positions (indices 1..N), plus inserting-substrate
# pairs: `nascent_OMP` = (BamA position, nascent position, effective radius, ideal distance),
# `nascent_LPS` = (LptD position, nascent position, ideal distance)
function make_agents(; OmpA=[], tethered_OmpA=[], OmpCF=[], LptD=[], tethered_LptD=[], BamA=[],
                       LPS=[], nascent_OMP=[], nascent_LPS=[])
    ix = Ref(0)
    next!() = (ix[] += 1)
    OmpAs = [M.OmpAAgent(next!(), p, 0.0, false, p) for p in OmpA]
    append!(OmpAs, [M.OmpAAgent(next!(), p, 0.0, true, t) for (p, t) in tethered_OmpA])
    OmpCFs = [M.OmpCFAgent(next!(), p, 0.0) for p in OmpCF]
    LptDs = [M.LptDAgent(next!(), p, 0.0, false, p, false, "free") for p in LptD]
    append!(LptDs, [M.LptDAgent(next!(), p, 0.0, true, t, true, "free") for (p, t) in tethered_LptD])
    BamAs = [M.BamAAgent(next!(), p, 0.0, false, "free") for p in BamA]
    LPSs = [M.LPSAgent(next!(), p, 0.0) for p in LPS]
    n_OMPs = M.NascentOMPAgent[]
    for (p_ins, p_sub, eff_rad, ideal) in nascent_OMP
        ins = M.BamAAgent(next!(), p_ins, 0.0, true, "embedding")
        push!(BamAs, ins)
        push!(n_OMPs, M.NascentOMPAgent(next!(), p_sub, 0.0, eff_rad, ideal, "OmpA", ins.index))
    end
    n_LPSs = M.NascentLPSAgent[]
    for (p_ins, p_sub, ideal) in nascent_LPS
        ins = M.LptDAgent(next!(), p_ins, 0.0, true, p_ins, true, "embedding")
        push!(LptDs, ins)
        push!(n_LPSs, M.NascentLPSAgent(next!(), p_sub, 0.0, ideal, ins.index))
    end
    return M.AllAgents(M.AllOMPs(OmpAs, OmpCFs, LptDs, BamAs), LPSs, M.AllNascent(n_OMPs, n_LPSs), 0)
end

# random system on an L x L domain, with untethered agents, tethered agents and nascent pairs
function random_agents(L; n_LPS=60, n_OMP=4, n_tethered=2, n_nascent=2)
    rp() = SVector(L * rand(), L * rand())
    near(p, d) = mod.(p + d * normalize(SVector(randn(), randn())), L)
    tethered(r) = [(p = rp(); (near(p, r * rand()), p)) for _ in 1:n_tethered]
    return make_agents(
        OmpA = [rp() for _ in 1:n_OMP], tethered_OmpA = tethered(0.02),
        OmpCF = [rp() for _ in 1:n_OMP],
        LptD = [rp() for _ in 1:n_OMP], tethered_LptD = tethered(0.05),
        BamA = [rp() for _ in 1:n_OMP],
        LPS = [rp() for _ in 1:n_LPS],
        nascent_OMP = [(p = rp(); (p, near(p, 0.3 * rand()), 0.05 * rand(), 0.25 * rand())) for _ in 1:n_nascent],
        nascent_LPS = [(p = rp(); (p, near(p, 0.3 * rand()), 0.25 * rand())) for _ in 1:n_nascent],
    )
end

# grid and flat data for `agents` on an L x L domain (as in initialise_system_random)
function setup_system(params, agents, L)
    grid_size, grid = M.initialise_grid(params)
    grid_size.dims = SVector(L, L)
    grid_size.num_agents = length(agent_list(agents))
    flat = M.initialise_flat_cpu(params)
    M.rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    M.compile_flat_system_data_cpu!(flat, agents, grid_size, grid, params)
    M.put_grid_in_sorted_order!(grid_size, grid, flat)
    return grid_size, grid, flat
end

# inserting-substrate partners, from the agent structs: sorted ix => (partner sorted ix, ideal distance)
function partners(agents, flat)
    d = Dict{Int, Tuple{Int, Float64}}()
    for n in Iterators.flatten((agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
        s_sub = sorted(flat, n)
        s_ins = flat.agent_ix_to_sorted_ix[n.inserting_agent_index]
        d[s_sub] = (s_ins, n.ideal_dist_from_inserting_agent)
        d[s_ins] = (s_sub, n.ideal_dist_from_inserting_agent)
    end
    return d
end
partner(parts, s) = haskey(parts, s) ? parts[s][1] : -1

# reference attraction-repulsion tally: sum of pairwise forces over *all* other agents
# (no grid), skipping the inserting-substrate partner
function brute_force_tally(flat, s, partner_s, params, n, dims)
    is_LPS(i) = M.parse_identifier(flat.identifiers[i])[2] == "LPS"
    pf = params.force
    F = SVector(0.0, 0.0)
    for j in 1:n
        (j == s || j == partner_s) && continue
        n_LPS = is_LPS(s) + is_LPS(j)
        mu_attr = n_LPS == 2 ? pf.mu_attr_LPS_LPS : n_LPS == 1 ? pf.mu_attr_OMP_LPS : pf.mu_attr_OMP_OMP
        F += M.compute_attr_rep_force(flat_pos(flat, s), flat.effective_radii[s],
            flat_pos(flat, j), flat.effective_radii[j], dims, pf.sensing_radius,
            mu_attr, pf.mu_rep, pf.max_repulsion, pf.rho, pf.k_C)
    end
    return F
end

# total force on sorted agent `s` (attraction-repulsion plus inserting-substrate)
function total_force(s, parts, grid_size, grid, flat, params)
    max_agg = maximum(flat.agg_dist_since_grid_sync)
    F = M.tally_attr_rep_forces(s, partner(parts, s), grid_size, grid, flat, params, max_agg)
    if haskey(parts, s)
        p, ideal = parts[s]
        ins = params.insertion
        F += M.compute_inserting_substrate_force(flat_pos(flat, s), flat_pos(flat, p), grid_size.dims,
            ideal, ins.mu_attr, ins.mu_rep, params.force.max_repulsion, params.force.rho, ins.k_C)
    end
    return F
end

# next position of sorted agent `s`, from a fresh state (compute_next_position mutates agg_dist)
function next_position(s, grid_size, grid, flat, params)
    return M.compute_next_position(s, flat, grid_size, grid, params, maximum(flat.agg_dist_since_grid_sync))
end


@testset "resolve_forces_cpu!" begin
    params = test_params()
    L = 3.0
    dims = SVector(L, L)

    # synchronous update: every agent moves as if computed alone from the pre-step state
    for _ in 1:3
        agents = random_agents(L)
        grid_size, grid, flat = setup_system(params, agents, L)
        n = grid_size.num_agents
        ref_grid_size, ref_grid, ref_flat = deepcopy((grid_size, grid, flat))
        expected = [next_position(s, ref_grid_size, ref_grid, deepcopy(ref_flat), params) for s in 1:n]

        M.resolve_forces_cpu!(agents, grid_size, grid, flat, params)
        @test all(flat_pos(flat, s) == expected[s] for s in 1:n)

        # agent structs and flat data agree, and everything is inside the domain
        @test all(a.position == flat_pos(flat, sorted(flat, a)) for a in agent_list(agents))
        @test flat.next_positions[1:2n] == flat.positions[1:2n]
        @test all(all(0 .<= a.position .< dims) for a in agent_list(agents))
    end

    # one step for a pair of LPS at separation `d`, returns (displacement 1, displacement 2)
    function step_pair(d)
        p1, p2 = SVector(1.5, 1.5), SVector(1.5 + d, 1.5)
        agents = make_agents(LPS=[p1, p2])
        grid_size, grid, flat = setup_system(params, agents, L)
        M.resolve_forces_cpu!(agents, grid_size, grid, flat, params)
        return agents.LPS[1].position - p1, agents.LPS[2].position - p2
    end
    ideal = 2 * params.LPS.radius

    # isolated agents don't move
    @test step_pair(ideal + params.force.sensing_radius + 0.1) == (SVector(0.0, 0.0), SVector(0.0, 0.0))

    # overlapping pair separates symmetrically
    d1, d2 = step_pair(0.7 * ideal)
    @test d1[1] < 0 && d2[1] > 0
    @test d1 ≈ -d2

    # pair in attraction range moves together symmetrically
    d1, d2 = step_pair(ideal + 0.5 * params.force.sensing_radius)
    @test d1[1] > 0 && d2[1] < 0
    @test d1 ≈ -d2
end


@testset "compile_flat_system_data_cpu!" begin
    params = test_params()
    L = 3.0

    for _ in 1:3
        agents = random_agents(L)
        grid_size, grid, flat = setup_system(params, agents, L)
        n = grid_size.num_agents

        # sorted <-> agent index maps are inverse permutations of 1:n
        @test sort(flat.agent_ix_to_sorted_ix[1:n]) == 1:n
        @test all(flat.sorted_ix_to_agent_ix[flat.agent_ix_to_sorted_ix[i]] == i for i in 1:n)

        # sorted order follows Morton cell order
        @test issorted(grid.agent_cell_z_ixs[1:n])

        # per-agent data: position, radius, type and tethering
        function check_agent(a, type_name, radius; tethered=false)
            s = sorted(flat, a)
            is_tethered, agent_type, _, _, _ = M.parse_identifier(flat.identifiers[s])
            ok = flat_pos(flat, s) == a.position && flat.effective_radii[s] == radius &&
                 agent_type == type_name && is_tethered == tethered
            tethered && (ok &= SVector(flat.tether_points[2s-1], flat.tether_points[2s]) == a.tether_point)
            return ok
        end
        @test all(check_agent(a, "OmpA", params.OmpA.radius; tethered=a.is_tethered) for a in agents.OMP.OmpA)
        @test all(check_agent(a, "OmpCF", params.OmpCF.radius) for a in agents.OMP.OmpCF)
        @test all(check_agent(a, "LptD", params.LptD.radius; tethered=a.is_tethered) for a in agents.OMP.LptD)
        @test all(check_agent(a, "BamA", params.BamA.radius) for a in agents.OMP.BamA)
        @test all(check_agent(a, "LPS", params.LPS.radius) for a in agents.LPS)
        @test all(check_agent(a, a.OMP_type, a.effective_radius) for a in agents.nascent.nascent_OMP)
        @test all(check_agent(a, "LPS", params.LPS.radius) for a in agents.nascent.nascent_LPS)

        # inserting-substrate bookkeeping
        for nascent in Iterators.flatten((agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
            s_sub = sorted(flat, nascent)
            s_ins = flat.agent_ix_to_sorted_ix[nascent.inserting_agent_index]
            _, _, sub_nascent, sub_inserting, k = M.parse_identifier(flat.identifiers[s_sub])
            _, _, ins_nascent, ins_inserting, k_ins = M.parse_identifier(flat.identifiers[s_ins])
            @test sub_nascent && !sub_inserting
            @test ins_inserting && !ins_nascent
            @test k_ins == k
            @test flat.nascent_to_substrate_ixs[k] == s_sub
            @test flat.nascent_to_inserting_ixs[k] == s_ins
            @test flat.substrate_inserting_ideal_dists[k] == nascent.ideal_dist_from_inserting_agent
        end

        # rebuilding resets the aggregated distances and is otherwise idempotent
        # (isequal, since unused buffer slots may hold NaN)
        ref = deepcopy(flat)
        flat.agg_dist_since_grid_sync .= 1.0
        M.compile_flat_system_data_cpu!(flat, agents, grid_size, grid, params)
        @test all(flat.agg_dist_since_grid_sync .== 0.0)
        for f in fieldnames(M.AllAgentsFlat)
            @test isequal(getfield(flat, f), getfield(ref, f))
        end

        # storage grows when the flat structure starts too small
        empty_flat = M.AllAgentsFlat(Float64[], Float64[], Float64[], Int[], Float64[], Float64[],
                                     Int[], Int[], Float64[], Int[], Int[])
        M.compile_flat_system_data_cpu!(empty_flat, agents, grid_size, grid, params)
        n_nascent = length(agents.nascent.nascent_OMP) + length(agents.nascent.nascent_LPS)
        for (f, len) in ((:positions, 2n), (:identifiers, n), (:effective_radii, n), (:tether_points, 2n),
                         (:agent_ix_to_sorted_ix, n), (:nascent_to_substrate_ixs, n_nascent),
                         (:substrate_inserting_ideal_dists, n_nascent))
            @test getfield(empty_flat, f)[1:len] == getfield(ref, f)[1:len]
        end
    end
end


@testset "compute_next_position" begin
    params = test_params()
    L = 3.0
    dims = SVector(L, L)

    # isolated agent: no force, no movement, no aggregated distance
    agents = make_agents(LPS=[SVector(1.5, 1.5)])
    grid_size, grid, flat = setup_system(params, agents, L)
    @test next_position(1, grid_size, grid, flat, params) == SVector(1.5, 1.5)
    @test flat.agg_dist_since_grid_sync[1] == 0.0

    # random systems (no tethers, so no moves are rejected)
    params_2dt = test_params(system=(dt=2 * params.system.dt,))
    params_2eta = test_params(force=(eta=2 * params.force.eta,))
    for _ in 1:3
        agents = random_agents(L; n_tethered=0)
        grid_size, grid, flat = setup_system(params, agents, L)
        parts = partners(agents, flat)
        flat_2dt, flat_2eta = deepcopy(flat), deepcopy(flat)

        for s in 1:grid_size.num_agents
            p = flat_pos(flat, s)
            F = total_force(s, parts, grid_size, grid, flat, params)
            next = next_position(s, grid_size, grid, flat, params)
            disp = M.shortest_vec(p, next, dims)

            # moves along the resultant force, stays inside the domain, records the distance moved
            @test abs(disp[1] * F[2] - disp[2] * F[1]) < 1e-12
            @test dot(disp, F) >= 0
            @test all(0 .<= next .< dims)
            @test flat.agg_dist_since_grid_sync[s] ≈ norm(disp) atol=1e-14

            # displacement linear in dt, inversely proportional to eta
            disp_2dt = M.shortest_vec(p, next_position(s, grid_size, grid, flat_2dt, params_2dt), dims)
            disp_2eta = M.shortest_vec(p, next_position(s, grid_size, grid, flat_2eta, params_2eta), dims)
            @test disp_2dt ≈ 2 * disp atol=1e-14
            @test disp_2eta ≈ disp / 2 atol=1e-14
        end
    end

    # nascent substrate with no other neighbours moves toward its inserting agent when beyond
    # its ideal distance, away when inside it
    p_ins, p_sub = SVector(1.5, 1.5), SVector(1.6, 1.5)
    for (ideal, toward) in ((0.05, true), (0.15, false))
        agents = make_agents(nascent_LPS=[(p_ins, p_sub, ideal)])
        grid_size, grid, flat = setup_system(params, agents, L)
        next = next_position(sorted(flat, agents.nascent.nascent_LPS[1]), grid_size, grid, flat, params)
        @test (norm(next - p_ins) < norm(p_sub - p_ins)) == toward
    end

    # nascent OMP mobility uses its actual radius, not its (growing) effective radius:
    # with only its inserting partner nearby, the effective radius doesn't change its motion
    moves = map((0.01, 0.04)) do eff_rad
        agents = make_agents(nascent_OMP=[(p_ins, p_sub, eff_rad, 0.05)])
        grid_size, grid, flat = setup_system(params, agents, L)
        next_position(sorted(flat, agents.nascent.nascent_OMP[1]), grid_size, grid, flat, params) - p_sub
    end
    @test norm(moves[1]) > 0
    @test moves[1] == moves[2]

    # tethered agents: a move that would leave the tether radius is rejected, otherwise accepted
    for (kw, tether_radius) in ((:tethered_OmpA, params.OmpA.tether_radius),
                                (:tethered_LptD, params.LptD.tether_radius))
        tether = SVector(1.5, 1.5)
        for (offset, accepted) in ((0.0, true), (tether_radius - 1e-6, false))
            p = tether + SVector(offset, 0.0)
            # overlapping LPS on the tether side pushes the agent in +x
            agents = make_agents(; kw => [(p, tether)], LPS=[p - SVector(0.05, 0.0)])
            grid_size, grid, flat = setup_system(params, agents, L)
            s = sorted(flat, agent_list(agents)[1])
            next = next_position(s, grid_size, grid, flat, params)
            @test (next != p) == accepted
            @test (flat.agg_dist_since_grid_sync[s] > 0) == accepted
            accepted && @test next[1] > p[1]
        end
    end
end


@testset "tally_attr_rep_forces" begin
    # distinct mu_attr per interaction type, so a wrong selection is visible
    params = test_params(force=(mu_attr_OMP_OMP=0.01, mu_attr_OMP_LPS=0.03, mu_attr_LPS_LPS=0.5))
    L = 3.0
    dims = SVector(L, L)
    tally(s, parts, grid_size, grid, flat) = M.tally_attr_rep_forces(s, partner(parts, s), grid_size, grid,
        flat, params, maximum(flat.agg_dist_since_grid_sync))

    for _ in 1:3
        agents = random_agents(L; n_LPS=150)
        grid_size, grid, flat = setup_system(params, agents, L)
        n = grid_size.num_agents
        parts = partners(agents, flat)

        # neighbour search finds exactly the agents a brute-force sum over all pairs does
        # (including across the periodic boundary, and skipping inserting-substrate partners)
        forces = [tally(s, parts, grid_size, grid, flat) for s in 1:n]
        @test all(forces[s] ≈ brute_force_tally(flat, s, partner(parts, s), params, n, dims) for s in 1:n)

        # Newton's third law: no net force on the whole system
        @test norm(sum(forces)) < 1e-12

        # periodic translation of the whole system leaves each agent's force unchanged
        shift = SVector(L * rand(), L * rand())
        for a in agent_list(agents)
            a.position = mod.(a.position + shift, L)
            a isa Union{M.OmpAAgent, M.LptDAgent} && (a.tether_point = mod.(a.tether_point + shift, L))
        end
        grid_size_2, grid_2, flat_2 = setup_system(params, agents, L)
        parts_2 = partners(agents, flat_2)
        for a in agent_list(agents)
            s, s2 = sorted(flat, a), sorted(flat_2, a)
            @test tally(s2, parts_2, grid_size_2, grid_2, flat_2) ≈ forces[s] atol=1e-12
        end

        # agents moved since the last grid sync are still found, given their aggregated distances
        for s in 1:n
            step = 0.05 * rand() * normalize(SVector(randn(), randn()))
            new_pos = mod.(flat_pos(flat_2, s) + step, L)
            flat_2.positions[2s-1], flat_2.positions[2s] = new_pos
            flat_2.agg_dist_since_grid_sync[s] = norm(step)
        end
        @test all(tally(s, parts_2, grid_size_2, grid_2, flat_2) ≈
                  brute_force_tally(flat_2, s, partner(parts_2, s), params, n, dims) for s in 1:n)
    end

    # isolated agent feels nothing
    agents = make_agents(LPS=[SVector(0.5, 0.5), SVector(2.0, 2.0)])
    grid_size, grid, flat = setup_system(params, agents, L)
    @test tally(1, Dict(), grid_size, grid, flat) == SVector(0.0, 0.0)

    # search window covering the whole grid visits each cell once: two large LptD on a small
    # domain, for an even (1 x 1, 10 cells) and odd (0.95 x 0.95, 9 cells) number of cells
    for L_small in (1.0, 0.95)
        agents = make_agents(LptD=[SVector(0.25, 0.5), SVector(0.72, 0.5)])
        grid_size, grid, flat = setup_system(params, agents, L_small)
        for s in 1:2
            @test tally(s, Dict(), grid_size, grid, flat) ≈
                  brute_force_tally(flat, s, -1, params, 2, SVector(L_small, L_small))
        end
    end
end


@testset "compute_attr_rep_force" begin
    p = SVector(0.5, 0.5)

    # zero beyond sensing range, and at (near-)zero separation
    @test attr_rep_force(p, p + SVector(IDEAL + FP.sensing_radius + 1e-6, 0.0)) == SVector(0.0, 0.0)
    @test attr_rep_force(p, p) == SVector(0.0, 0.0)
    @test attr_rep_force(p, p + SVector(1e-9, 0.0)) == SVector(0.0, 0.0)

    # saturated repulsion inside rho*ideal
    for d in range(1e-6, FP.rho * IDEAL; length=10)
        @test signed_mag(d) ≈ FP.max_repulsion
    end

    # never more repulsive than max_repulsion; attractive outside contact
    for d in range(1e-6, IDEAL + FP.sensing_radius; length=500)
        @test signed_mag(d) >= FP.max_repulsion - 1e-12
        d > IDEAL && @test signed_mag(d) >= 0
    end

    # continuous at contact distance, from both sides
    ϵ = 1e-9
    @test abs(signed_mag(IDEAL + ϵ)) < 1e-6
    @test abs(signed_mag(IDEAL - ϵ)) < 1e-6

    # symmetry and periodic boundaries
    check_symmetry(attr_rep_force, IDEAL + FP.sensing_radius)
    check_wrap_attraction(attr_rep_force, IDEAL + 0.005)

    # zero attraction (cf. config TEST_zero_OMP_OMP_attraction): no force outside contact,
    # repulsion unaffected
    for d in range(IDEAL, IDEAL + FP.sensing_radius; length=200)
        @test signed_mag(d; mu_attr=0.0) == 0.0
    end
    @test signed_mag(0.8 * IDEAL; mu_attr=0.0) == signed_mag(0.8 * IDEAL)
end


@testset "compute_inserting_substrate_force" begin
    p = SVector(0.5, 0.5)

    # zero when the ideal distance is (near-)zero, e.g. a newly created nascent agent
    for ideal in (0.0, 1e-9), d in (1e-3, 0.5 * IDEAL, IDEAL, 3 * IDEAL)
        @test ins_force(p, p + SVector(d, 0.0); ideal) == SVector(0.0, 0.0)
    end

    # zero at (near-)zero separation
    @test ins_force(p, p) == SVector(0.0, 0.0)
    @test ins_force(p, p + SVector(1e-9, 0.0)) == SVector(0.0, 0.0)

    # saturated repulsion inside rho*ideal
    for d in range(1e-6, FP.rho * IDEAL; length=10)
        @test ins_signed_mag(d) ≈ FP.max_repulsion
    end

    # never more repulsive than max_repulsion; attractive outside contact
    for d in range(1e-6, 4 * IDEAL; length=500)
        @test ins_signed_mag(d) >= FP.max_repulsion - 1e-12
        d > IDEAL && @test ins_signed_mag(d) >= 0
    end

    # no sensing-radius cutoff: still attractive well beyond where the pairwise force vanishes
    for d in (IDEAL + FP.sensing_radius + 1e-3, 2 * IDEAL, 3 * IDEAL)
        @test ins_signed_mag(d) > 0
    end

    # continuous at contact distance, from both sides
    ϵ = 1e-9
    @test abs(ins_signed_mag(IDEAL + ϵ)) < 1e-6
    @test abs(ins_signed_mag(IDEAL - ϵ)) < 1e-6

    # symmetry and periodic boundaries
    check_symmetry(ins_force, 3 * IDEAL)
    check_wrap_attraction(ins_force, 2 * IDEAL)

    # growing ideal distance at fixed separation: attraction turns into repulsion once
    # the ideal distance exceeds the separation (the substrate is pushed out as it grows)
    d = 0.05
    @test ins_signed_mag(d; ideal=0.5 * d) > 0
    @test ins_signed_mag(d; ideal=1.5 * d) < 0
end


@testset "compute_diffusion!" begin
    L = 3.0
    dims = SVector(L, L)

    # positions of all agents after one diffusion step, alongside the starting positions
    function diffuse(params; agents=random_agents(L))
        grid_size, grid, flat = setup_system(params, agents, L)
        before = [a.position for a in agent_list(agents)]
        M.compute_diffusion!(agents, flat, grid_size, params)
        return agents, flat, before
    end
    moved(agents, before) = [a.position != b for (a, b) in zip(agent_list(agents), before)]

    # switched off: nothing moves
    for force in ((temperature=0.0, diffusion_mode="LPS_only"), (temperature=0.01, diffusion_mode="off"))
        agents, _, before = diffuse(test_params(; force))
        @test !any(moved(agents, before))
    end

    # LPS only: LPS and nascent LPS move, OMPs don't; flat data kept in sync
    agents, flat, before = diffuse(test_params(force=(temperature=0.01, diffusion_mode="LPS_only")))
    is_LPS = [a isa Union{M.LPSAgent, M.NascentLPSAgent} for a in agent_list(agents)]
    @test moved(agents, before) == is_LPS
    for (a, b) in zip(agent_list(agents), before)
        s = sorted(flat, a)
        @test all(0 .<= a.position .< dims)
        @test flat_pos(flat, s) == a.position
        @test flat.agg_dist_since_grid_sync[s] ≈ norm(M.shortest_vec(b, a.position, dims)) atol=1e-14
    end

    # Brownian statistics (seeded): zero mean, isotropic, variance linear in dt and increasing
    # with temperature
    function LPS_steps(; temperature=0.01, dt=1e-4, n_reps=40)
        params = test_params(system=(; dt), force=(; temperature, diffusion_mode="LPS_only"))
        agents = make_agents(LPS=[SVector(L * rand(), L * rand()) for _ in 1:500])
        grid_size, _, flat = setup_system(params, agents, L)
        steps = SVector{2, Float64}[]
        for _ in 1:n_reps
            before = [a.position for a in agents.LPS]
            M.compute_diffusion!(agents, flat, grid_size, params)
            append!(steps, [M.shortest_vec(b, a.position, dims) for (a, b) in zip(agents.LPS, before)])
        end
        return first.(steps), last.(steps)
    end
    Random.seed!(1234)
    x, y = LPS_steps()
    n = length(x)
    var_x, var_y = sum(abs2, x) / n, sum(abs2, y) / n
    @test abs(sum(x) / n) < 5 * sqrt(var_x / n)
    @test abs(sum(y) / n) < 5 * sqrt(var_y / n)
    @test isapprox(var_x, var_y; rtol=0.05)

    x2, y2 = LPS_steps(dt=2e-4)
    @test isapprox((sum(abs2, x2) + sum(abs2, y2)) / n, 2 * (var_x + var_y); rtol=0.05)

    x3, y3 = LPS_steps(temperature=0.02)
    @test (sum(abs2, x3) + sum(abs2, y3)) / n > 1.5 * (var_x + var_y)

    # OMP diffusion is no longer supported
    @test_throws ErrorException diffuse(test_params(force=(temperature=0.01, diffusion_mode="all")))
end
