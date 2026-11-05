# Unit tests for the functions in src/initialise.jl, in source order:
#   initialise_system_cpu, initialise_grid, initialise_flat_cpu, initialise_agents_random,
#   initialise_agents_specify_positions, build_nascent_added_area_lookup, compute_nascent_incs,
#   initialise_system_random, initialise_system_specify_positions,
#   initialise_system_from_checkpoint, assemble_all_agents!, check_params
#
# Not yet covered (they run dynamics): run_equilibration!, run_membrane_shrinkage!


# small demo system, no equilibration, output to a fresh temporary directory
# (joinpath("out", abs_path) == abs_path, so nothing is written into the repo)
const SMALL_INIT = (num_OmpA=3, num_OmpCF=2, num_LptD=2, num_BamA=2, num_LPS=15, num_PP=4,
                    equilibration_time=0.0)
small_params(; init=(;), system=(;), kw...) =
    test_params(; init=(; SMALL_INIT..., init...), system=(; output_dir=mktempdir(), system...), kw...)

n_agents(params) = params.init.num_OmpA + params.init.num_OmpCF + params.init.num_LptD +
                   params.init.num_BamA + params.init.num_LPS

type_name(a::M.NascentOMPAgent) = a.OMP_type
type_name(a::Union{M.LPSAgent, M.NascentLPSAgent}) = "LPS"
type_name(a::M.AbstractOMP) = Dict(M.OmpAAgent => "OmpA", M.OmpCFAgent => "OmpCF",
                                   M.LptDAgent => "LptD", M.BamAAgent => "BamA")[typeof(a)]

# agent counts per type match the parameters
function check_counts(agents, params)
    @test length(agents.OMP.OmpA) == params.init.num_OmpA
    @test length(agents.OMP.OmpCF) == params.init.num_OmpCF
    @test length(agents.OMP.LptD) == params.init.num_LptD
    @test length(agents.OMP.BamA) == params.init.num_BamA
    @test length(agents.LPS) == params.init.num_LPS
end

# agents, grid size and flat data describe the same system
function check_consistent(agents, grid_size, flat)
    all_agents = agent_list(agents)
    n = grid_size.num_agents
    @test length(all_agents) == n
    @test sort([a.index for a in all_agents]) == 1:n
    @test all(all(0 .<= a.position .< grid_size.dims) for a in all_agents)
    @test all(flat_pos(flat, sorted(flat, a)) == a.position for a in all_agents)
    @test all(M.parse_identifier(flat.identifiers[sorted(flat, a)])[2] == type_name(a) for a in all_agents)
end

# all fields of two agent lists are equal
same_agents(as, bs) = length(as) == length(bs) && all(
    typeof(a) == typeof(b) && all(getfield(a, f) == getfield(b, f) for f in fieldnames(typeof(a)))
    for (a, b) in zip(as, bs))

# OmpA/LptD tethered at their own positions (flat data too), BamA/LptD assembled
function check_assembled(agents, flat)
    for a in Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD))
        s = sorted(flat, a)
        @test a.is_tethered && a.tether_point == a.position
        @test SVector(flat.tether_points[2s-1], flat.tether_points[2s]) == a.position
        @test M.parse_identifier(flat.identifiers[s])[1]
    end
    @test all(a.is_part_of_assembled_complex for a in Iterators.flatten((agents.OMP.BamA, agents.OMP.LptD)))
end


@testset "initialise_system_cpu" begin
    params = small_params()
    agents, grid_size, grid, flat = M.initialise_system_cpu(params; suppress_prints=true)

    @test isdir(joinpath(params.system.output_dir, "raw_data"))
    check_counts(agents, params)
    check_consistent(agents, grid_size, flat)

    @test_throws ErrorException M.initialise_system_cpu(small_params(init=(method="bogus",)); suppress_prints=true)
end


@testset "initialise_grid" begin
    params = small_params(init=(dim_x=2.0, dim_y=2.0))
    grid_size, grid = M.initialise_grid(params)

    # empty grid of the right size, flagged as changed so it is built on the first update
    @test grid_size.dims == SVector(2.0, 2.0)
    @test grid_size.num_cells == SVector(0, 0) && grid_size.tot_num_cells == 0
    @test grid_size.num_agents == n_agents(params)
    @test grid_size.grid_size_changed && grid_size.num_agents_changed && !grid_size.nascent_promoted
    @test all(isempty(getfield(grid, f)) for f in fieldnames(M.SimGrid))

    # first rebuild puts every agent in a cell
    agents = M.initialise_agents_random(grid_size, params)
    M.rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    @test grid_size.tot_num_cells > 0
    @test sum(grid.num_agents_in_cell[1:grid_size.tot_num_cells]) == grid_size.num_agents
end


@testset "initialise_flat_cpu" begin
    params = small_params()
    n = n_agents(params)
    flat = M.initialise_flat_cpu(params)

    for f in (:positions, :next_positions, :tether_points)
        @test length(getfield(flat, f)) == 2n
    end
    for f in (:effective_radii, :identifiers, :agg_dist_since_grid_sync, :agent_ix_to_sorted_ix,
              :sorted_ix_to_agent_ix)
        @test length(getfield(flat, f)) == n
    end
    @test length(flat.nascent_to_inserting_ixs) == length(flat.nascent_to_substrate_ixs) ==
          length(flat.substrate_inserting_ideal_dists)
end


@testset "initialise_agents_random" begin
    params = small_params(init=(dim_x=2.0, dim_y=2.0))
    grid_size, _ = M.initialise_grid(params)
    agents = M.initialise_agents_random(grid_size, params)
    all_agents = agent_list(agents)

    check_counts(agents, params)
    @test sort([a.index for a in all_agents]) == 1:n_agents(params)
    @test all(all(0 .<= a.position .< grid_size.dims) for a in all_agents)

    # everything starts untethered, unassembled and free, with no insertions under way
    @test !any(a.is_tethered for a in Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD)))
    @test all(!a.is_part_of_assembled_complex && a.insertion_state == "free"
              for a in Iterators.flatten((agents.OMP.LptD, agents.OMP.BamA)))
    @test isempty(agents.nascent.nascent_OMP) && isempty(agents.nascent.nascent_LPS)
    @test agents.num_PP == params.init.num_PP
end


# random positions well inside the 1 x 1 demo domain
rand_positions(n) = [SVector(0.1 + 0.8 * rand(), 0.1 + 0.8 * rand()) for _ in 1:n]

# demo params with "specify_positions" initialisation (positions given per type, or nothing)
function positions_params(; OmpA=nothing, OmpCF=nothing, LptD=nothing, BamA=nothing, LPS=nothing, init=(;))
    return small_params(init=(; method="specify_positions",
                              positions=M.InitPositions(OmpA, OmpCF, LptD, BamA, LPS), init...))
end

positions_of(agents) = (OmpA=[a.position for a in agents.OMP.OmpA], OmpCF=[a.position for a in agents.OMP.OmpCF],
                        LptD=[a.position for a in agents.OMP.LptD], BamA=[a.position for a in agents.OMP.BamA],
                        LPS=[a.position for a in agents.LPS])

@testset "initialise_agents_specify_positions" begin
    full = (OmpA=rand_positions(3), OmpCF=rand_positions(2), LptD=rand_positions(2),
            BamA=rand_positions(2), LPS=rand_positions(15))
    place(params) = M.initialise_agents_specify_positions(M.initialise_grid(params)[1], params)

    # all positions given: used exactly, no random placement
    params = positions_params(; full...)
    agents, random_yn = place(params)
    @test !random_yn
    @test positions_of(agents) == full
    @test sort([a.index for a in agent_list(agents)]) == 1:n_agents(params)
    @test !any(a.is_tethered for a in Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD)))
    @test all(a.insertion_state == "free" for a in Iterators.flatten((agents.OMP.LptD, agents.OMP.BamA)))

    # some positions missing: given ones used, the rest placed at random in the domain
    params = positions_params(; full..., OmpCF=nothing, LPS=full.LPS[1:5])
    agents, random_yn = place(params)
    @test random_yn
    check_counts(agents, params)
    @test positions_of(agents).OmpA == full.OmpA
    @test positions_of(agents).LPS[1:5] == full.LPS[1:5]
    @test all(all(0 .<= a.position .< SVector(1.0, 1.0)) for a in agent_list(agents))

    # a type with no agents needs no positions
    _, random_yn = place(positions_params(; full..., OmpCF=nothing, init=(num_OmpCF=0,)))
    @test !random_yn

    # invalid input
    @test_throws ErrorException place(small_params(init=(method="specify_positions", positions=nothing)))
    @test_throws AssertionError place(positions_params(; full..., OmpA=rand_positions(4)))
    @test_throws AssertionError place(positions_params(; full..., OmpA=vcat([SVector(-0.1, 0.5)], full.OmpA[2:end])))
    @test_throws AssertionError place(positions_params(; full..., OmpA=vcat([SVector(0.5, 1.5)], full.OmpA[2:end])))
end


@testset "build_nascent_added_area_lookup" begin
    params = small_params(system=(dt=1e-3,))
    dt = params.system.dt
    rate = params.insertion.OMP_assembly_rate
    lookup = M.build_nascent_added_area_lookup(params)

    # (table, substrate radius, insertion time)
    tables = ((lookup.OmpA, params.OmpA.radius, params.OmpA.radius / rate),
              (lookup.OmpCF, params.OmpCF.radius, params.OmpCF.radius / rate),
              (lookup.BamA, params.BamA.radius, params.BamA.radius / rate),
              (lookup.LptD, params.LptD.radius, params.LptD.radius / rate),
              (lookup.LPS, params.LPS.radius, params.LPS.insertion_time))
    for (table, r_sub, T) in tables
        # no area at the start, area only ever added
        @test table[1] == 0.0
        @test all(table .>= -1e-15)

        # over the whole insertion, the area added is exactly the substrate's own area
        @test sum(table) ≈ π * r_sub^2 rtol=1e-3

        # long enough to look up every step of an insertion (cf. compute_added_area)
        @test length(table) >= floor(Int, T / dt) + 1
    end

    # no OMP assembly: OMP tables skipped (only allowed if no polypeptides can bind)
    params_0 = small_params(system=(dt=1e-3,), insertion=(OMP_assembly_rate=0.0, PP_arrival_rate=0.0))
    lookup_0 = redirect_stdout(() -> M.build_nascent_added_area_lookup(params_0), devnull)
    @test all(isempty, (lookup_0.OmpA, lookup_0.OmpCF, lookup_0.BamA, lookup_0.LptD))
    @test lookup_0.LPS == lookup.LPS
    @test_throws AssertionError M.build_nascent_added_area_lookup(
        small_params(insertion=(OMP_assembly_rate=0.0,)))
end


@testset "compute_nascent_incs" begin
    # one BamA and one LptD to insert from
    base = small_params(init=(num_OmpA=0, num_OmpCF=0, num_LptD=1, num_BamA=1, num_LPS=0),
                        system=(dt=1e-3,))
    dt = base.system.dt
    rate = base.insertion.OMP_assembly_rate
    eff_incs, dist_incs = M.compute_nascent_incs(base)

    radius_of = Dict("OmpA" => base.OmpA.radius, "OmpCF" => base.OmpCF.radius,
                     "BamA" => base.BamA.radius, "LptD" => base.LptD.radius)

    # a nascent OMP of each type starts hidden inside BamA, and after its insertion time
    # has grown to full size, just touching BamA
    for (OMP_type, radius) in radius_of
        params = base
        for t in keys(radius_of)
            prob = t == OMP_type ? 1.0 : 0.0
            params = M.change_field(params, t, with_fields(getfield(params, Symbol(t)); insertion_prob=prob))
        end
        agents, grid_size, _, flat = M.initialise_system_random(params; suppress_prints=true)
        BamA = agents.OMP.BamA[1]
        M.generate_nascent_OMP_obj!(agents, grid_size, flat, BamA, 0.0, params)
        nascent = agents.nascent.nascent_OMP[end]
        @test nascent.OMP_type == OMP_type

        # (large OMPs start a tiny offset, insertion_eps, from BamA's centre)
        @test M.exposed_area(params.BamA.radius, nascent.effective_radius,
                             nascent.ideal_dist_from_inserting_agent) < 1e-4 * π * radius^2

        for _ in 1:round(Int, radius / rate / dt)
            M.update_nascent_OMP!(nascent, eff_incs, dist_incs)
        end
        k = Dict("OmpA" => 1, "OmpCF" => 3, "BamA" => 5, "LptD" => 7)[OMP_type]
        @test nascent.effective_radius ≈ radius atol=abs(eff_incs[k]) + 1e-9
        @test nascent.ideal_dist_from_inserting_agent ≈ params.BamA.radius + radius atol=abs(dist_incs[k]) + 1e-5
    end

    # nascent LPS: never changes size, starts inside LptD, ends just touching it
    @test eff_incs[2] == 0.0
    agents, grid_size, _, _ = M.initialise_system_random(base; suppress_prints=true)
    M.generate_nascent_LPS_obj!(agents, grid_size, agents.OMP.LptD[1], 0.0, base)
    nascent = agents.nascent.nascent_LPS[end]
    @test M.exposed_area(base.LptD.radius, base.LPS.radius, nascent.ideal_dist_from_inserting_agent) ≈ 0.0 atol=1e-12
    for _ in 1:round(Int, base.LPS.insertion_time / dt)
        nascent.ideal_dist_from_inserting_agent += dist_incs[2]
    end
    @test nascent.ideal_dist_from_inserting_agent ≈ base.LptD.radius + base.LPS.radius atol=abs(dist_incs[2]) + 1e-9
end


@testset "initialise_system_random" begin
    for assembled in (true, false, nothing)
        params = small_params(init=(complexes_assembled=assembled,))
        agents, grid_size, grid, flat = M.initialise_system_random(params; suppress_prints=true)

        check_counts(agents, params)
        check_consistent(agents, grid_size, flat)
        @test !grid_size.num_agents_changed

        if assembled == true
            check_assembled(agents, flat)
        else
            @test !any(a.is_tethered for a in Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD)))
            @test !any(a.is_part_of_assembled_complex for a in Iterators.flatten((agents.OMP.BamA, agents.OMP.LptD)))
        end
    end
end


@testset "initialise_system_specify_positions" begin
    # all positions given (including an overlapping pair): agents stay exactly where placed,
    # even with a nonzero equilibration time
    full = (OmpA=rand_positions(3), OmpCF=rand_positions(2), LptD=rand_positions(2),
            BamA=rand_positions(2), LPS=rand_positions(15))
    full.LPS[2] = full.LPS[1] + SVector(0.01, 0.0)
    params = positions_params(; full..., init=(equilibration_time=1.0,))
    agents, grid_size, grid, flat = M.initialise_system_specify_positions(params; suppress_prints=true)

    @test positions_of(agents) == full
    check_consistent(agents, grid_size, flat)
    @test !grid_size.num_agents_changed
    check_assembled(agents, flat)    # demo config has complexes_assembled = true
end


@testset "initialise_system_from_checkpoint" begin
    # save a system with an insertion under way
    params = small_params()
    agents, grid_size, _, _ = M.initialise_system_random(params; suppress_prints=true)
    M.generate_nascent_LPS_obj!(agents, grid_size, agents.OMP.LptD[1], 0.0, params)
    out_dir = mktempdir()
    mkpath(joinpath(out_dir, "raw_data"))
    M.write_system_state(agents, grid_size.dims, out_dir, 7)
    file = joinpath(out_dir, "raw_data", "sys_data_7.jld2")

    # reload: domain size comes from the checkpoint, not the config
    cp_params = small_params(init=(method="from_checkpoint", checkpoint_file=file, dim_x=5.0, dim_y=5.0))
    loaded, grid_size_2, _, flat_2 = M.initialise_system_from_checkpoint(cp_params; suppress_prints=true)

    @test grid_size_2.dims == grid_size.dims
    @test grid_size_2.num_agents == n_agents(params) + 1
    @test !grid_size_2.num_agents_changed
    @test same_agents(agent_list(loaded), agent_list(agents))
    check_consistent(loaded, grid_size_2, flat_2)

    # the nascent LPS is linked to its LptD in the flat data
    nascent = loaded.nascent.nascent_LPS[1]
    k = M.parse_identifier(flat_2.identifiers[sorted(flat_2, nascent)])[5]
    @test flat_2.nascent_to_inserting_ixs[k] == sorted(flat_2, loaded.OMP.LptD[1])

    # missing checkpoint
    @test_throws AssertionError M.initialise_system_from_checkpoint(
        small_params(init=(method="from_checkpoint", checkpoint_file=nothing)); suppress_prints=true)
    @test_throws AssertionError M.initialise_system_from_checkpoint(
        small_params(init=(method="from_checkpoint", checkpoint_file=joinpath(out_dir, "missing.jld2")));
        suppress_prints=true)
end


@testset "assemble_all_agents!" begin
    params = small_params(init=(complexes_assembled=false,))
    agents, grid_size, grid, flat = M.initialise_system_random(params; suppress_prints=true)
    ids_before = copy(flat.identifiers)

    M.assemble_all_agents!(agents, flat)
    check_assembled(agents, flat)

    # other agents' flat data untouched
    for a in Iterators.flatten((agents.OMP.OmpCF, agents.OMP.BamA, agents.LPS))
        @test flat.identifiers[sorted(flat, a)] == ids_before[sorted(flat, a)]
    end

    # assembling again changes nothing
    agents_ref, flat_ref = deepcopy((agents, flat))
    M.assemble_all_agents!(agents, flat)
    @test same_agents(agent_list(agents), agent_list(agents_ref))
    @test flat.identifiers == flat_ref.identifiers
    @test isequal(flat.tether_points, flat_ref.tether_points)
end


@testset "check_params" begin
    no_attraction = (mu_attr_OMP_OMP=0.0, mu_attr_OMP_LPS=0.0, mu_attr_LPS_LPS=0.0)

    @test_logs M.check_params(test_params(force=no_attraction))
    @test_logs (:warn, r"LPS-LPS") M.check_params(test_params())     # demo LPS-LPS attraction is strong
    @test_logs (:warn, r"OMP-LPS") M.check_params(test_params(force=(; no_attraction..., mu_attr_OMP_LPS=100.0)))
    @test_logs (:warn, r"OMP-OMP") M.check_params(test_params(force=(; no_attraction..., mu_attr_OMP_OMP=100.0)))
end
