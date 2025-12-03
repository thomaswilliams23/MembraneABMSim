


"""
    initialise_system(params::AllParams; clear_existing_output::Bool=false)

Initialises a simulation based on the parameter structure passed in. Depending on the initialisation
type specified in the parameter structure, calls a helper function to assemble the system.

Output:
(
    agents::AllAgents,
    grid_size::GridSize,
    grid::SimGrid,
    system_flat::AllAgentsFlat
)

"""
function initialise_system_cpu(params::AllParams; clear_existing_output::Bool=false)

    # set up output directory structure
    set_up_output_directory(params.system.output_dir; clear_existing_output=clear_existing_output)

    # decide which initialisation to use
    if params.init.method == "random"
        println("Initialising model with random distribution of agents...")
        agents, grid_size, grid, system_flat_cpu = initialise_system_random(params)
    else
        #TODO: implement other initialisation methods?
        error("Initialisation type $(params.initialisation.init_type) not recognised.")
    end

    # return the initialised model
    return (agents, grid_size, grid, system_flat_cpu)
end


"""
    initialise_grid(params::AllParams)

Initialises a spatial grid for the simulation based on the parameters provided.
"""
function initialise_grid(params::AllParams)
    
    # get dimensions and cell size from params
    dims = SVector{2, Float64}(params.init.dim_x, params.init.dim_y)
    
    #initial number of cells - this is zero for initialisation, gets filled on first update
    num_cells_init = SVector{2, Int}(0, 0)

    #initial number of agents
    num_agents_init = (
        params.init.num_BamA + 
        params.init.num_OmpA + 
        params.init.num_OmpCF + 
        params.init.num_LptD + 
        params.init.num_LPS
    )

    grid_size_changed_init = true
    num_agents_changed_init = true
    nascent_promoted_init = false

    #grid_size object
    grid_size = GridSize(
        dims,
        num_cells_init,
        prod(num_cells_init),
        num_agents_init,
        grid_size_changed_init,
        num_agents_changed_init,
        nascent_promoted_init
    )


    #make grid object (mostly empty for now, gets filled on first update)
    coords_to_z_ix = Int[]
    z_ix_to_coords = Int[]
    agent_cell_z_ixs_init = Int[]
    agent_ixs_in_cell_init = Vector{Int}[]
    num_agents_in_cell = Int[]
    start_agents_in_cell = Int[]

    grid =  SimGrid(
        coords_to_z_ix,
        z_ix_to_coords,
        agent_cell_z_ixs_init, 
        agent_ixs_in_cell_init,
        num_agents_in_cell,
        start_agents_in_cell,
    )

    return (
        grid_size,
        grid
    )
end


"""
    initialise_flat_cpu(params::AllParams)

Initialises a blank flat data structure used for force calculation.
"""
function initialise_flat_cpu(params::AllParams) :: AllAgentsFlat

    #number of agents
    num_agents_init = (
        params.init.num_BamA + 
        params.init.num_OmpA + 
        params.init.num_OmpCF + 
        params.init.num_LptD + 
        params.init.num_LPS
    )

    #initial buffer for nascent-inserting agents
    nascent_buffer_size = round(Int, 1.5 * (params.init.num_BamA + params.init.num_LptD))

    #allocate memory
    positions = Vector{Float64}(undef, 2*num_agents_init)
    next_positions = Vector{Float64}(undef, 2*num_agents_init)
    effective_radii = Vector{Float64}(undef, num_agents_init)
    identifiers = Vector{Int}(undef, num_agents_init)
    tether_points = Vector{Float64}(undef, 2*num_agents_init)
    agg_dist_since_grid_sync = Vector{Float64}(undef, num_agents_init)
    nascent_to_inserting_ixs = Vector{Int}(undef, nascent_buffer_size)
    nascent_to_substrate_ixs = Vector{Int}(undef, nascent_buffer_size)
    substrate_inserting_ideal_dists = Vector{Float64}(undef, nascent_buffer_size)
    agent_ix_to_sorted_ix = Vector{Int}(undef, num_agents_init)
    sorted_ix_to_agent_ix = Vector{Int}(undef, num_agents_init)

    return AllAgentsFlat(
        positions,
        next_positions,
        effective_radii,
        identifiers,
        tether_points,
        agg_dist_since_grid_sync,
        nascent_to_inserting_ixs,
        nascent_to_substrate_ixs,
        substrate_inserting_ideal_dists,
        agent_ix_to_sorted_ix,
        sorted_ix_to_agent_ix
    )
end



"""
    initialise_agents_random(grid_size::GridSize, params::AllParams)

Initialises the AllAgents structure with agents randomly placed within the domain.
"""
function initialise_agents_random(grid_size::GridSize, params::AllParams)

    #initialisation defaults
    init_agent_arrival_time = -Inf     #init objects assumed to have arrived at -Inf

    is_tethered_init = false           #all agents initialised as untethered
    tether_point_init = 0.0            #all agents initialised as untethered
    is_assembled_init = false          #all agents initialised as unassembled
    insertion_state_init = "free"      #all agents initialised in the "free" state

    nascent_OMPs_init = Vector{NascentOMPAgent}()
                                       #no initial nascent objects
    nascent_LPS_init = Vector{NascentLPSAgent}()
                                       #no initial nascent objects

    #initialise agents and package them
    num_agents_allocated_so_far = 0
    all_OmpAs = [
        OmpAAgent(
            ix + num_agents_allocated_so_far,
            SVector{2, Float64}(rand() * grid_size.dims[1], rand() * grid_size.dims[2]),
            init_agent_arrival_time, 
            is_tethered_init, 
            SVector{2, Float64}(tether_point_init, tether_point_init)
        )
        for ix in 1:params.init.num_OmpA
    ]

    num_agents_allocated_so_far += params.init.num_OmpA
    all_OmpCFs = [
        OmpCFAgent(
            ix + num_agents_allocated_so_far,
            SVector{2, Float64}(rand() * grid_size.dims[1], rand() * grid_size.dims[2]),
            init_agent_arrival_time
        )
        for ix in 1:params.init.num_OmpCF
    ]

    num_agents_allocated_so_far += params.init.num_OmpCF
    all_LptDs = [
        LptDAgent(
            ix + num_agents_allocated_so_far,
            SVector{2, Float64}(rand() * grid_size.dims[1], rand() * grid_size.dims[2]),
            init_agent_arrival_time, 
            is_tethered_init, 
            SVector{2, Float64}(tether_point_init, tether_point_init), 
            is_assembled_init, 
            insertion_state_init
        )
        for ix in 1:params.init.num_LptD
    ]

    num_agents_allocated_so_far += params.init.num_LptD
    all_BamAs = [
        BamAAgent(
            ix + num_agents_allocated_so_far,
            SVector{2, Float64}(rand() * grid_size.dims[1], rand() * grid_size.dims[2]),
            init_agent_arrival_time, 
            is_assembled_init, 
            insertion_state_init
        )
        for ix in 1:params.init.num_BamA
    ]

    num_agents_allocated_so_far += params.init.num_BamA
    all_LPS = [
        LPSAgent(
            ix + num_agents_allocated_so_far,
            SVector{2, Float64}(rand() * grid_size.dims[1], rand() * grid_size.dims[2]),
            init_agent_arrival_time
        )
        for ix in 1:params.init.num_LPS
    ]

    num_PP_init = params.init.num_PP
    agents = AllAgents(
        AllOMPs(all_OmpAs, all_OmpCFs, all_LptDs, all_BamAs),
        all_LPS,
        AllNascent(nascent_OMPs_init, nascent_LPS_init),
        num_PP_init
    )

    return agents
end




"""
    build_nascent_added_area_lookup(params::AllParams)

Builds a vector for each of the agent types, listing the area it adds over the process of its 
insertion.
"""
function build_nascent_added_area_lookup(params::AllParams)

    time_err = 0.001*params.system.dt

    #initialise
    OmpA_added_area = Float64[]
    OmpCF_added_area = Float64[]
    BamA_added_area = Float64[]
    LptD_added_area = Float64[]
    LPS_added_area = Float64[]

    #skip OMPs if assembly rate is zero
    assembly_rate_err = 1e-8
    if params.insertion.OMP_assembly_rate<assembly_rate_err
        @assert (params.insertion.PP_arrival_rate<assembly_rate_err) || (params.insertion.PP_bind_rate<assembly_rate_err)
        println("OMP insertion rate is zero. Skipping construction of nascent added area lookup table for OMPs.")
    else

        #OmpA
        init_rad = min(params.OmpA.radius, params.BamA.radius)
        final_rad = params.OmpA.radius
        init_dist = params.BamA.radius - init_rad
        final_dist = params.BamA.radius + final_rad
        insertion_time = params.OmpA.radius/params.insertion.OMP_assembly_rate

        push!(OmpA_added_area, 0.0) #no area added initially
        prev_exposed_area = 0.0
        time_elapsed = params.system.dt
        while time_elapsed < insertion_time + time_err
            curr_rad = init_rad + (time_elapsed/insertion_time) * (final_rad - init_rad)
            curr_dist = init_dist + (time_elapsed/insertion_time) * (final_dist - init_dist)

            curr_exposed_area = exposed_area(params.BamA.radius, curr_rad, curr_dist)
            added_area = curr_exposed_area - prev_exposed_area
            push!(OmpA_added_area, added_area)

            prev_exposed_area = curr_exposed_area
            time_elapsed += params.system.dt
        end


        #OmpCF
        init_rad = min(params.OmpCF.radius, params.BamA.radius)
        final_rad = params.OmpCF.radius
        init_dist = params.BamA.radius - init_rad
        final_dist = params.BamA.radius + final_rad
        insertion_time = params.OmpCF.radius/params.insertion.OMP_assembly_rate

        push!(OmpCF_added_area, 0.0) #no area added initially
        prev_exposed_area = 0.0
        time_elapsed = params.system.dt
        while time_elapsed < insertion_time + time_err
            curr_rad = init_rad + (time_elapsed/insertion_time) * (final_rad - init_rad)
            curr_dist = init_dist + (time_elapsed/insertion_time) * (final_dist - init_dist)

            curr_exposed_area = exposed_area(params.BamA.radius, curr_rad, curr_dist)
            added_area = curr_exposed_area - prev_exposed_area
            push!(OmpCF_added_area, added_area)

            prev_exposed_area = curr_exposed_area
            time_elapsed += params.system.dt
        end


        #BamA
        init_rad = params.BamA.radius
        final_rad = params.BamA.radius
        init_dist = params.BamA.radius - init_rad
        final_dist = params.BamA.radius + final_rad
        insertion_time = params.BamA.radius/params.insertion.OMP_assembly_rate

        push!(BamA_added_area, 0.0) #no area added initially
        prev_exposed_area = 0.0
        time_elapsed = params.system.dt
        while time_elapsed < insertion_time + time_err
            curr_rad = init_rad + (time_elapsed/insertion_time) * (final_rad - init_rad)
            curr_dist = init_dist + (time_elapsed/insertion_time) * (final_dist - init_dist)

            curr_exposed_area = exposed_area(params.BamA.radius, curr_rad, curr_dist)
            added_area = curr_exposed_area - prev_exposed_area
            push!(BamA_added_area, added_area)

            prev_exposed_area = curr_exposed_area
            time_elapsed += params.system.dt
        end


        #LptD
        init_rad = min(params.LptD.radius, params.BamA.radius)
        final_rad = params.LptD.radius
        init_dist = params.BamA.radius - init_rad
        final_dist = params.BamA.radius + final_rad
        insertion_time = params.LptD.radius/params.insertion.OMP_assembly_rate

        push!(LptD_added_area, 0.0) #no area added initially
        prev_exposed_area = 0.0
        time_elapsed = params.system.dt
        while time_elapsed < insertion_time + time_err
            curr_rad = init_rad + (time_elapsed/insertion_time) * (final_rad - init_rad)
            curr_dist = init_dist + (time_elapsed/insertion_time) * (final_dist - init_dist)

            curr_exposed_area = exposed_area(params.BamA.radius, curr_rad, curr_dist)
            added_area = curr_exposed_area - prev_exposed_area
            push!(LptD_added_area, added_area)

            prev_exposed_area = curr_exposed_area
            time_elapsed += params.system.dt
        end
    end


    #LPS
    init_dist = params.LptD.radius - params.LPS.radius
    final_dist = params.LptD.radius + params.LPS.radius
    insertion_time = params.LPS.insertion_time

    push!(LPS_added_area, 0.0) #no area added initially
    prev_exposed_area = 0.0
    time_elapsed = params.system.dt
    while time_elapsed < insertion_time + time_err
        curr_dist = init_dist + (time_elapsed/insertion_time) * (final_dist - init_dist)

        curr_exposed_area = exposed_area(params.LptD.radius, params.LPS.radius, curr_dist)
        added_area = curr_exposed_area - prev_exposed_area
        push!(LPS_added_area, added_area)

        prev_exposed_area = curr_exposed_area
        time_elapsed += params.system.dt
    end


    return NascentAddedAreaLookup(
        OmpA_added_area,
        OmpCF_added_area,
        BamA_added_area,
        LptD_added_area,
        LPS_added_area
    )

end


"""
    compute_nascent_incs(params::AllParams)

Precompute increments for effective radius and distance increments for nascent agents 
as they are inserted.
"""
function compute_nascent_incs(params::AllParams)
    #initialise
    effective_rad_incs = Vector{Float64}(undef, 7)
    ideal_dist_incs = Vector{Float64}(undef, 7)

    #maps
    agent_type_to_num = Dict(
        "OmpA" => 1,
        "OmpCF" => 3,
        "BamA" => 5,
        "LptD" => 7,
        "LPS" => 2
    )
    agent_type_to_rad = Dict(
        "OmpA" => params.OmpA.radius,
        "OmpCF" => params.OmpCF.radius,
        "BamA" => params.BamA.radius,
        "LptD" => params.LptD.radius,
        "LPS" => params.LPS.radius
    )

    #helper
    function _compute_insertion_incs_this_OMP_type(agent_type::String)
        final_rad = agent_type_to_rad[agent_type]
        final_dist = params.BamA.radius + final_rad
        if final_rad>params.BamA.radius
            init_rad = params.BamA.radius
            init_dist = 0.0
        else
            init_rad = final_rad
            init_dist = params.BamA.radius - final_rad
        end
        insertion_time = final_rad/params.insertion.OMP_assembly_rate
        effective_rad_inc = (params.system.dt/insertion_time)*(final_rad - init_rad)
        ideal_dist_inc = (params.system.dt/insertion_time)*(final_dist - init_dist)
        return (effective_rad_inc, ideal_dist_inc)
    end

    #compute increments for each agent type
    for agent_type in keys(agent_type_to_num)
        #OMP
        if agent_type != "LPS"
            agent_type_num = agent_type_to_num[agent_type]
            (effective_rad_incs[agent_type_num], ideal_dist_incs[agent_type_num]) = _compute_insertion_incs_this_OMP_type(agent_type)
            continue
        end
        #LPS
        agent_type_num = 2
        final_dist = params.LptD.radius + params.LPS.radius
        init_dist = params.LptD.radius - params.LPS.radius
        insertion_time = params.LPS.insertion_time
        effective_rad_incs[agent_type_num] = 0.0  #LPS does not change size during insertion
        ideal_dist_incs[agent_type_num] = (params.system.dt/insertion_time)*(final_dist - init_dist)
    end

    return (
        SVector{7, Float64}(effective_rad_incs), 
        SVector{7, Float64}(ideal_dist_incs)
    )
end



"""
    initialise_system_random(params::AllParams)

Initialises a simulation with agents randomly placed within the domain. Runs equilibration 
to resolve forces.
"""
function initialise_system_random(params::AllParams)

    #initialise simulation objects
    grid_size, grid = initialise_grid(params)
    system_flat_cpu = initialise_flat_cpu(params)
    agents = initialise_agents_random(grid_size, params)

    #populate data structures
    rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
    put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)
    grid_size.num_agents_changed = false

    #run equilibration
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #temporary
    t = 0.0
    time_err = 0.1 * params.system.dt
    while t < params.init.equilibration_time - time_err

        if steps_since_grid_sync >= MAX_STEPS_BETWEEN_GRID_SYNC
            rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
            compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
            put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

            steps_since_grid_sync = 0
        else
            steps_since_grid_sync += 1
        end

        resolve_forces_cpu!(agents, grid_size, grid, system_flat_cpu, params)
        
        t += params.system.dt


        #report time
        if abs(t/params.system.vis_dt - round(t/params.system.vis_dt))<time_err
            @printf "Running equilibration: τ=%5.2f\r" t
        end
    end
    println("\nEquilibration complete.")

    #if specified, set all agents as assembled/tethered
    if params.init.complexes_assembled
        assemble_all_agents!(agents, system_flat_cpu)
    end

    return (agents, grid_size, grid, system_flat_cpu)
end



"""
    assemble_all_agents!(agents::AllAgents)

If this option is specified in the parameters structure, make all relevant agents tethered or 
assembled.
"""
function assemble_all_agents!(agents::AllAgents, system_flat::AllAgentsFlat)
    for OmpA in agents.OMP.OmpA
        OmpA.is_tethered = true
        OmpA.tether_point = OmpA.position
        #change the flat data structure too
        sorted_ix = system_flat.agent_ix_to_sorted_ix[OmpA.index]
        system_flat.tether_points[2*(sorted_ix-1)+1] = OmpA.tether_point[1]
        system_flat.tether_points[2*(sorted_ix-1)+2] = OmpA.tether_point[2]
        system_flat.identifiers[sorted_ix] = -abs(system_flat.identifiers[sorted_ix])
    end
    for BamA in agents.OMP.BamA
        BamA.is_part_of_assembled_complex = true
    end
    for LptD in agents.OMP.LptD
        LptD.is_part_of_assembled_complex = true
        LptD.is_tethered = true
        LptD.tether_point = LptD.position
        #change the flat data structure too
        sorted_ix = system_flat.agent_ix_to_sorted_ix[LptD.index]
        system_flat.tether_points[2*(sorted_ix-1)+1] = LptD.tether_point[1]
        system_flat.tether_points[2*(sorted_ix-1)+2] = LptD.tether_point[2]
        system_flat.identifiers[sorted_ix] = -abs(system_flat.identifiers[sorted_ix])
    end
end




"""
    check_params(params::AllParams)

Checks the numerical stability of the parameters provided.
"""
function check_params(params::AllParams)

    max_OMP_radius = max(params.OmpA.radius, params.OmpCF.radius, params.LptD.radius, params.BamA.radius)


    #check if attraction is too strong relative to maximum repulsion
    MAX_ATTR_TO_REP_RATIO = 0.3

    #LPS-LPS
    max_LPS_LPS_attraction = (params.force.mu_attr_LPS_LPS * 2*params.LPS.radius) / (params.force.k_C * exp(1))
    if max_LPS_LPS_attraction / abs(params.force.max_repulsion) > MAX_ATTR_TO_REP_RATIO
        @warn "LPS-LPS attraction strength may be too high relative to maximum repulsion and may result in overlaps. Consider increasing max repulsion and decreasing the time step."
    end

    #OMP-LPS
    max_OMP_LPS_attraction = (params.force.mu_attr_OMP_LPS * (max_OMP_radius + params.LPS.radius)) / (params.force.k_C * exp(1))
    if max_OMP_LPS_attraction / abs(params.force.max_repulsion) > MAX_ATTR_TO_REP_RATIO
        @warn "OMP-LPS attraction strength may be too high relative to maximum repulsion and may result in overlaps. Consider increasing max repulsion and decreasing the time step."
    end

    #OMP-OMP
    max_OMP_OMP_attraction = (params.force.mu_attr_OMP_OMP * 2*max_OMP_radius) / (params.force.k_C * exp(1))
    if max_OMP_OMP_attraction / abs(params.force.max_repulsion) > MAX_ATTR_TO_REP_RATIO
        @warn "OMP-OMP attraction strength may be too high relative to maximum repulsion and may result in overlaps. Consider increasing max repulsion and decreasing the time step."
    end


end