


"""
    initialise_system(params::AllParams)

Initialises a simulation based on the parameter structure passed in. Depending on the initialisation
type specified in the parameter structure, calls a helper function to assemble the system.

Output:
(
    agents::AllAgents,
    grid::SimGrid,
    system_flat::AllAgentsFlat,
    nascent_added_area_lookup::NascentAddedAreaLookup
)

"""
function initialise_system(params::AllParams)

    # set up output directory structure
    set_up_output_directory(params.system.output_dir)

    #make a lookup table for area additions from nascent agents
    nascent_added_area_lookup = build_nascent_added_area_lookup(params)

    # decide which initialisation to use
    if params.init.method == "random"
        println("Initialising model with random distribution of agents...")
        agents, grid, system_flat = initialise_system_random(params)
    else
        #TODO: implement other initialisation methods?
        error("Initialisation type $(params.initialisation.init_type) not recognised.")
    end

    # return the initialised model (placeholder for now)
    return (agents, grid, system_flat, nascent_added_area_lookup)
end


"""
    initialise_grid(params::AllParams) -> SimGrid

Initialises a spatial grid for the simulation based on the parameters provided.
"""
function initialise_grid(params::AllParams) :: SimGrid
    
    # get dimensions and cell size from params
    dims = SVector{2, Float64}(params.init.dim_x, params.init.dim_y)
    cell_width = params.force.sensing_radius
    num_cells = SVector{2, Int}(ceil.(Int, dims./cell_width))

    #make initial mappings between coords and Morton z index
    coords_to_z_ix, z_ix_to_coords = build_dense_morton(num_cells)

    # create empty cells
    start_agent_init = nothing
    num_agents_init = 0
    cells_init = Vector{SimCell}(undef, prod(num_cells))
    for cell_z_ix = 1:prod(num_cells)
        agent_ixs_init = Int[]
        cells_init[cell_z_ix] = SimCell(
            cell_z_ix,
            start_agent_init,
            num_agents_init,
            agent_ixs_init
        )
    end

    #other info we need to set up the grid
    num_agents_init = (
        params.init.num_BamA + 
        params.init.num_OmpA + 
        params.init.num_OmpCF + 
        params.init.num_LptD + 
        params.init.num_LPS
    )
    agent_cell_z_ixs_init = zeros(Int, num_agents_init)

    return SimGrid(
        dims,
        num_cells,
        num_agents_init,  
        coords_to_z_ix,
        z_ix_to_coords,
        agent_cell_z_ixs_init, 
        cells_init
    )
end


"""
    initialise_flat(params::AllParams) -> AllAgentsFlat

Initialises a blank flat data structure used for force calculation.
"""
function initialise_flat(params::AllParams) :: AllAgentsFlat
    num_agents_init = (
        params.init.num_BamA + 
        params.init.num_OmpA + 
        params.init.num_OmpCF + 
        params.init.num_LptD + 
        params.init.num_LPS
    )
    positions = Vector{Float64}(undef, 2*num_agents_init)
    next_positions = Vector{Float64}(undef, 2*num_agents_init)
    effective_radii = Vector{Float64}(undef, num_agents_init)
    actual_radii = Vector{Float64}(undef, num_agents_init)
    is_OMP = Vector{Bool}(undef, num_agents_init)
    is_tethered = Vector{Bool}(undef, num_agents_init)
    tether_points = Vector{Float64}(undef, 2*num_agents_init)
    tether_lengths = Vector{Float64}(undef, num_agents_init)
    substrate_inserting_ixs = Vector{Int}(undef, num_agents_init)
    substrate_inserting_ideal_dists = Vector{Float64}(undef, num_agents_init)
    successors = Vector{Int}()
    agent_ix_to_sorted_ix = Vector{Int}()
    sorted_ix_to_agent_ix = Vector{Int}()

    return AllAgentsFlat(
        positions,
        next_positions,
        effective_radii,
        actual_radii,
        is_OMP,
        is_tethered,
        tether_points,
        tether_lengths,
        substrate_inserting_ixs,
        substrate_inserting_ideal_dists,
        successors,
        agent_ix_to_sorted_ix,
        sorted_ix_to_agent_ix
    )
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
    initialise_system_random(params::AllParams)

Initialises a simulation with agents randomly placed within the domain. Runs equilibration 
to resolve positions.
"""
function initialise_system_random(params::AllParams)

    #initialise spatial grid and flat structure for force calculation
    grid = initialise_grid(params)
    system_flat = initialise_flat(params)

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
            SVector{2, Float64}(rand() * grid.dims[1], rand() * grid.dims[2]),
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
            SVector{2, Float64}(rand() * grid.dims[1], rand() * grid.dims[2]),
            init_agent_arrival_time
        )
        for ix in 1:params.init.num_OmpCF
    ]

    num_agents_allocated_so_far += params.init.num_OmpCF
    all_LptDs = [
        LptDAgent(
            ix + num_agents_allocated_so_far,
            SVector{2, Float64}(rand() * grid.dims[1], rand() * grid.dims[2]),
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
            SVector{2, Float64}(rand() * grid.dims[1], rand() * grid.dims[2]),
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
            SVector{2, Float64}(rand() * grid.dims[1], rand() * grid.dims[2]),
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
    
    #run equilibration
    t = 0.0
    time_err = 0.1 * params.system.dt
    while t < params.init.equilibration_time - time_err
        rebuild_grid!(grid, agents, params.force.sensing_radius)
        compile_flat_system_data!(system_flat, agents, grid, params)
        resolve_forces!(agents, grid, system_flat, params)
        t += params.system.dt
    end

    #if specified, set all agents as assembled/tethered
    if params.init.complexes_assembled
        assemble_all_agents!(agents)
    end

    return (agents, grid, system_flat)
end



"""
    function assemble_all_agents!(agents::AllAgents)

If this option is specified in the parameters structure, make all relevant agents tethered or 
assembled.
"""
function assemble_all_agents!(agents::AllAgents)
    for OmpA in agents.OMP.OmpA
        OmpA.is_tethered = true
        OmpA.tether_point = OmpA.position
    end
    for BamA in agents.OMP.BamA
        BamA.is_part_of_assembled_complex = true
    end
    for LptD in agents.OMP.LptD
        LptD.is_part_of_assembled_complex = true
        LptD.is_tethered = true
        LptD.tether_point = LptD.position
    end
end