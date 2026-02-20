


"""
    initialise_system(params::AllParams; clear_existing_output::Bool=false, suppress_prints::Bool=false)

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
function initialise_system_cpu(params::AllParams; clear_existing_output::Bool=false, suppress_prints::Bool=false)

    # set up output directory structure
    set_up_output_directory(params.system.output_dir; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints)

    # decide which initialisation to use
    if params.init.method == "random"
        if !suppress_prints
            println("Initialising model with random distribution of agents...")
        end
        agents, grid_size, grid, system_flat_cpu = initialise_system_random(params; suppress_prints=suppress_prints)
    elseif params.init.method == "specify_positions"
        if !suppress_prints
            println("Initialising model with specified agent positions...")
        end
        agents, grid_size, grid, system_flat_cpu = initialise_system_specify_positions(params; suppress_prints=suppress_prints)
    elseif params.init.method == "from_checkpoint"
        if !suppress_prints
            println("Initialising model by continuing from checkpoint...")
        end
        agents, grid_size, grid, system_flat_cpu = initialise_system_from_checkpoint(params; suppress_prints=suppress_prints)
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
    initialise_agents_specify_positions(grid_size::GridSize, params::AllParams)

Initialises the AllAgents structure with agents placed at specified positions within the domain.
Any positions not specified are randomly assigned.
"""
function initialise_agents_specify_positions(grid_size::GridSize, params::AllParams)

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

    #first check that positions have been specified
    if isnothing(params.init.positions)
        error("No positions specified in params.init.positions for 'specify_positions' initialisation method. Use 'random' method or provide positions.")
    end

    #check positions are valid
    for agent_type in [:OmpA, :OmpCF, :LptD, :BamA, :LPS]
        pos_list = getfield(params.init.positions, agent_type)
        num_agents_of_type = getfield(params.init, Symbol("num_$agent_type"))
        if !isnothing(pos_list)
            @assert length(pos_list) <= num_agents_of_type "Number of specsified positions for $agent_type exceeds number of agents of that type."
            for pos in pos_list
                @assert all(pos .>= 0.0) "Specified position $pos for $agent_type is out of bounds (negative coordinate)."
                @assert pos[1] <= grid_size.dims[1] "Specified x position $(pos[1]) for $agent_type exceeds domain size ($(grid_size.dims[1]))."
                @assert pos[2] <= grid_size.dims[2] "Specified y position $(pos[2]) for $agent_type exceeds domain size ($(grid_size.dims[2]))."
            end
        end
    end

    #check if any random placement is required
    includes_random_placement = false
    for agent_type in [:OmpA, :OmpCF, :LptD, :BamA, :LPS]
        pos_list = getfield(params.init.positions, agent_type)
        num_agents_of_type = getfield(params.init, Symbol("num_$agent_type"))
        if num_agents_of_type > 0
            if isnothing(pos_list)
                includes_random_placement = true
                break
            elseif length(pos_list) < num_agents_of_type
                includes_random_placement = true
                break
            end
        end
    end

    #helper to get position or random if not specified
    function _get_position_or_random(pos_list::Union{Nothing, Vector{SVector{2, Float64}}}, ix::Int)
        if isnothing(pos_list) || length(pos_list)<ix
            return SVector{2, Float64}(rand() * grid_size.dims[1], rand() * grid_size.dims[2])
        else
            return pos_list[ix]
        end
    end

    #initialise agents and package them
    num_agents_allocated_so_far = 0
    all_OmpAs = [
        OmpAAgent(
            ix + num_agents_allocated_so_far,
            _get_position_or_random(params.init.positions.OmpA, ix),
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
            _get_position_or_random(params.init.positions.OmpCF, ix),
            init_agent_arrival_time
        )
        for ix in 1:params.init.num_OmpCF
    ]

    num_agents_allocated_so_far += params.init.num_OmpCF
    all_LptDs = [
        LptDAgent(
            ix + num_agents_allocated_so_far,
            _get_position_or_random(params.init.positions.LptD, ix),
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
            _get_position_or_random(params.init.positions.BamA, ix),
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
            _get_position_or_random(params.init.positions.LPS, ix),
            init_agent_arrival_time
        )
        for ix in 1:params.init.num_LPS
    ]

    #group together
    num_PP_init = params.init.num_PP
    agents = AllAgents(
        AllOMPs(all_OmpAs, all_OmpCFs, all_LptDs, all_BamAs),
        all_LPS,
        AllNascent(nascent_OMPs_init, nascent_LPS_init),
        num_PP_init
    )

    return (agents, includes_random_placement)
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
        run_equilibration!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, params::AllParams, equilibration_time::Float64; suppress_prints::Bool=false)

Runs equilibration for a specified amount of time, updating the system at each timestep and synchronising the grid as needed. Used to resolve forces before starting the main simulation loop.
"""
function run_equilibration!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, params::AllParams, equilibration_time::Float64; suppress_prints::Bool=false)

    #run equilibration
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #temporary
    t = 0.0
    time_err = 0.1 * params.system.dt
    while t < equilibration_time - time_err

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
            if !suppress_prints
                @printf "Running equilibration: τ=%5.2f\r" t
            end
        end
    end
    if !suppress_prints
        println("\nEquilibration complete.")
    end
end



"""
    run_membrane_shrinkage!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, params::AllParams; suppress_prints::Bool=false)

Runs an iterative membrane shrinkage procedure to remove holes in the initial configuration. Shrinks the domain iteratively, equilibrating at each step, until holes are removed 
or a maximum number of shrinkage rounds is reached.
"""
function run_membrane_shrinkage!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, params::AllParams; suppress_prints::Bool=false)

    @assert !isnothing(params.system.max_hole_radius) "max_hole_radius must be specified in params.system to run membrane shrinkage."

    if !suppress_prints
        println("Shrinking domain to fit agent packing.")
    end

    #check if we even need to shrink the domain - if there are no holes, skip this step
    if !membrane_contains_hole(agents, grid_size, params)
        if !suppress_prints
            println("Membrane is already well-packed with no holes. Skipping shrinkage.")
        end
    else

        #decide the rate at which we shrink the domain - can be passed as a parameter
        DEFAULT_SHRINKAGE_FACTOR = 0.01
        shrinkage_factor = isnothing(params.init.shrinkage_factor) ? DEFAULT_SHRINKAGE_FACTOR : params.init.shrinkage_factor

        #shrink the domain iteratively until holes are removed - again, can be passed as a parameter
        DEFAULT_MAX_NUM_SHRINKAGE_ROUNDS = 10
        max_num_shrinkage_rounds = isnothing(params.init.max_num_shrinkage_rounds) ? DEFAULT_MAX_NUM_SHRINKAGE_ROUNDS : params.init.max_num_shrinkage_rounds

        #also the amount of time we spend on each round of shrinkage - can be passed as a parameter
        DEFAULT_SHRINKAGE_EQUILIBRATION_TIME = 3.0
        shrinkage_equilibration_time = isnothing(params.init.shrinkage_equilibration_time) ? DEFAULT_SHRINKAGE_EQUILIBRATION_TIME : params.init.shrinkage_equilibration_time

        if !suppress_prints
            println("Carrying out up to $max_num_shrinkage_rounds rounds of shrinkage with shrinkage factor $(100*shrinkage_factor)% and equilibration time of $shrinkage_equilibration_time per round.")
        end

        #do iterative shrinkage
        shrinkage_round_num = 1
        membrane_contains_hole_yn = true
        while membrane_contains_hole_yn && shrinkage_round_num <= max_num_shrinkage_rounds

            if !suppress_prints
                @printf "Running shrinkage round %d of %d\r" shrinkage_round_num max_num_shrinkage_rounds
            end

            #shrink the domain
            grid_size.dims *= (1.0 - shrinkage_factor)

            shrinkage_scale_factor = (1.0 - shrinkage_factor)
            apply_scale_factor!(agents, system_flat_cpu, grid_size, shrinkage_scale_factor)

            run_equilibration!(agents, grid_size, grid, system_flat_cpu, params, shrinkage_equilibration_time; suppress_prints=true)

            shrinkage_round_num += 1
            membrane_contains_hole_yn = membrane_contains_hole(agents, grid_size, params)
        end
        
        if membrane_contains_hole_yn
            @warn "\nMaximum number of shrinkage rounds reached but membrane still contains holes. Consider increasing the number of shrinkage rounds, increasing the shrinkage factor, or adjusting other initialisation parameters to achieve a better-packed initial configuration."
        else
            if !suppress_prints
                println("\nShrinkage complete. Membrane is now well-packed with no holes.")
            end
        end
    end
end




"""
    initialise_system_random(params::AllParams)

Initialises a simulation with agents randomly placed within the domain. Runs equilibration 
to resolve forces.
"""
function initialise_system_random(params::AllParams; suppress_prints::Bool=false)

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
    if params.init.equilibration_time > 0.0
        run_equilibration!(agents, grid_size, grid, system_flat_cpu, params, params.init.equilibration_time; suppress_prints=suppress_prints)
    end

    #optionally, shrink the domain to fit the agents after equilibration
    if params.init.shrink_to_size == true
        run_membrane_shrinkage!(agents, grid_size, grid, system_flat_cpu, params; suppress_prints=suppress_prints)
    end

    #if specified, set all agents as assembled/tethered
    if params.init.complexes_assembled == true
        assemble_all_agents!(agents, system_flat_cpu)
    end

    return (agents, grid_size, grid, system_flat_cpu)
end


"""
    initialise_system_specify_positions(params::AllParams; suppress_prints::Bool=false)

Initialises a simulation with agents placed at specified positions within the domain.
"""
function initialise_system_specify_positions(params::AllParams; suppress_prints::Bool=false)

    #initialise simulation objects
    grid_size, grid = initialise_grid(params)
    system_flat_cpu = initialise_flat_cpu(params)
    agents, includes_random_placement = initialise_agents_specify_positions(grid_size, params)

    #populate data structures
    rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
    put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)
    grid_size.num_agents_changed = false

    #if random placement was included, run equilibration
    if includes_random_placement
        if !suppress_prints
            println("Some agents were randomly placed. Running equilibration.")
        end

        run_equilibration!(agents, grid_size, grid, system_flat_cpu, params, params.init.equilibration_time; suppress_prints=suppress_prints)

        #optionally, shrink the domain to fit the agents after equilibration
        if params.init.shrink_to_size == true
            run_membrane_shrinkage!(agents, grid_size, grid, system_flat_cpu, params; suppress_prints=suppress_prints)
        end

    else
        if !suppress_prints
            println("All agents placed at specified positions. Skipping equilibration.")
        end
    end

    #if specified, set all agents as assembled/tethered
    if params.init.complexes_assembled == true
        assemble_all_agents!(agents, system_flat_cpu)
    end

    return (agents, grid_size, grid, system_flat_cpu)
end




"""
    initialise_system_from_checkpoint(params::AllParams; suppress_prints::Bool=false)

Initialises the simulation system by loading from a checkpoint file specified in the parameters.
"""
function initialise_system_from_checkpoint(params::AllParams; suppress_prints::Bool=false)

    #initialise the system
    grid_size, grid = initialise_grid(params)
    system_flat_cpu = initialise_flat_cpu(params)

    #load in agents and dims from checkpoint
    @assert !isnothing(params.init.checkpoint_file) "No checkpoint file specified in params.init.checkpoint_file for 'from_checkpoint' initialisation method."
    @assert isfile(params.init.checkpoint_file) "Checkpoint file specified in params.init.checkpoint_file does not exist: $(params.init.checkpoint_file)"
    if !suppress_prints
        println("Loading checkpoint file from: $(params.init.checkpoint_file)")
    end
    @load params.init.checkpoint_file agents dims

    #set grid dimensions
    grid_size.dims = dims
    grid_size.num_agents = (
        length(agents.OMP.OmpA) + 
        length(agents.OMP.OmpCF) + 
        length(agents.OMP.LptD) + 
        length(agents.OMP.BamA) + 
        length(agents.LPS) + 
        length(agents.nascent.nascent_OMP) + 
        length(agents.nascent.nascent_LPS)
    )

    #populate data structures
    rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
    put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)
    grid_size.num_agents_changed = false

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

Checks the parameters for potential problems, such as numerical problems or inconsistencies.
"""
function check_params(params::AllParams)

    #check if attraction is too strong relative to maximum repulsion
    MAX_ATTR_TO_REP_RATIO = 0.3
    max_OMP_radius = max(params.OmpA.radius, params.OmpCF.radius, params.LptD.radius, params.BamA.radius)

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

    #TODO: way more checks can be added here

end