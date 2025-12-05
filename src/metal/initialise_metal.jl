"""
    initialise_system_metal(params::AllParams; clear_existing_output::Bool=false, suppress_prints::Bool=false)

Initialises the simulation system using the Metal GPU backend. Includes additional data 
structures necessary for running the simulation with a GPU.
"""
function initialise_system_metal(params::AllParams; clear_existing_output::Bool=false, suppress_prints::Bool=false)

    #build the kernels
    non_force_position_kernel = _non_force_position_kernel_metal!(MetalBackend())
    force_kernel = _force_kernel_metal!(MetalBackend())

    # set up output directory structure
    set_up_output_directory(params.system.output_dir; clear_existing_output=clear_existing_output)

    # decide which initialisation to use
    if params.init.method == "random"
        if !suppress_prints
            println("Initialising model with random distribution of agents...")
        end
        (
            agents, 
            grid_size,
            grid,
            system_flat_cpu,
            all_data_metal,
            grid_size_metal,
            params_metal
        ) = initialise_system_random_metal(force_kernel, params; suppress_prints=suppress_prints)
    else
        #TODO: implement other initialisation methods?
        error("Initialisation type $(params.initialisation.init_type) not recognised.")
    end

    # return the initialised model
    return (non_force_position_kernel, force_kernel, agents, grid_size, grid, system_flat_cpu, all_data_metal, grid_size_metal, params_metal)
end



"""
    initialise_system_random_metal(force_kernel, params::AllParams)

Initialises agents randomly within the domain and runs equilibration using Metal GPU backend.
"""
function initialise_system_random_metal(force_kernel, params::AllParams; suppress_prints::Bool=false)

    #initialise as per CPU version
    grid_size, grid = initialise_grid(params)
    system_flat_cpu = initialise_flat_cpu(params)
    agents = initialise_agents_random(grid_size, params)

    #initialise CPU-bound metal data
    grid_size_metal, params_metal = initialise_CPU_metal_data(grid_size, params)

    #initialise GPU-bound data
    all_data_metal = initialise_GPU_metal_data(grid_size, params)
    
    #build initial grid
    rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
    put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

    #copy data to metal GPU
    copy_data_to_metal!(all_data_metal, grid_size_metal, system_flat_cpu, grid_size, grid)

    #run equilibration
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #temporary
    t = 0.0
    time_err = 0.1 * params.system.dt
    while t < params.init.equilibration_time - time_err

        if steps_since_grid_sync >= MAX_STEPS_BETWEEN_GRID_SYNC

            copy_data_to_cpu_from_metal!(system_flat_cpu, agents, all_data_metal, grid_size_metal)

            rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
            compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
            put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

            copy_data_to_metal!(all_data_metal, grid_size_metal, system_flat_cpu, grid_size, grid)

            steps_since_grid_sync = 0
        else
            steps_since_grid_sync += 1
        end

        resolve_forces_metal!(
            force_kernel, 
            all_data_metal, 
            grid_size_metal, 
            params_metal
        )
        KernelAbstractions.synchronize(MetalBackend())
        copyto!(all_data_metal.positions, all_data_metal.next_positions[1:2*grid_size_metal.num_agents])
        
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
    
    #now copy across to cpu
    if params.init.equilibration_time > time_err
        copy_data_to_cpu_from_metal!(system_flat_cpu, agents, all_data_metal, grid_size_metal)
    else
        #if no equilibration, still need to copy initial positions across
        copyto!(all_data_metal.next_positions, all_data_metal.positions[1:2*grid_size_metal.num_agents])
    end

    #if specified, set all agents as assembled/tethered
    if params.init.complexes_assembled
        assemble_all_agents!(agents, system_flat_cpu)

        #this will also change identifiers and tether points so copy across to GPU
        copyto!(all_data_metal.identifiers, system_flat_cpu.identifiers[1:grid_size.num_agents])
        copyto!(all_data_metal.tether_points, system_flat_cpu.tether_points[1:2*grid_size.num_agents])
    end

    return (agents, grid_size, grid, system_flat_cpu, all_data_metal, grid_size_metal, params_metal)
end




"""
    initialise_GPU_metal_data(grid_size::GridSize, params::AllParams)

Initialises GPU-bound data structures needed for running the simulation on a Metal backend.
"""
function initialise_GPU_metal_data(grid_size::GridSize, params::AllParams)

    #first build the flat system data
    num_agents_init = (
        params.init.num_BamA + 
        params.init.num_OmpA + 
        params.init.num_OmpCF + 
        params.init.num_LptD + 
        params.init.num_LPS
    )

    #initial buffer for nascent-inserting agents
    nascent_buffer_size = round(Int, 1.5 * (params.init.num_BamA + params.init.num_LptD))

    #initial buffer for newly tethered agents
    newly_tethered_buffer_size = 10

    #allocate memory on GPU
    positions = MtlVector{Float32, Metal.PrivateStorage}(undef, 2*num_agents_init)
    next_positions = MtlVector{Float32, Metal.PrivateStorage}(undef, 2*num_agents_init)
    effective_radii = MtlVector{Float32, Metal.PrivateStorage}(undef, num_agents_init)
    identifiers = MtlVector{Int, Metal.PrivateStorage}(undef, num_agents_init)
    tether_points = MtlVector{Float32, Metal.PrivateStorage}(undef, 2*num_agents_init)
    agg_dist_since_grid_sync = MtlVector{Float32, Metal.PrivateStorage}(undef, num_agents_init)

    nascent_to_inserting_ixs = MtlVector{Int, Metal.PrivateStorage}(undef, nascent_buffer_size)
    nascent_to_substrate_ixs = MtlVector{Int, Metal.PrivateStorage}(undef, nascent_buffer_size)
    substrate_inserting_ideal_dists = MtlVector{Float32, Metal.PrivateStorage}(undef, nascent_buffer_size)
    
    coords_to_z_ix = MtlVector{Int, Metal.PrivateStorage}(undef, grid_size.tot_num_cells)
    z_ix_to_coords = MtlVector{Int, Metal.PrivateStorage}(undef, 2*grid_size.tot_num_cells)
    agent_cell_z_ixs = MtlVector{Int, Metal.PrivateStorage}(undef, num_agents_init)
    num_agents_in_cell = MtlVector{Int, Metal.PrivateStorage}(undef, grid_size.tot_num_cells)
    start_agents_in_cell = MtlVector{Int, Metal.PrivateStorage}(undef, grid_size.tot_num_cells)

    newly_tethered_agent_ixs = MtlVector{Int, Metal.PrivateStorage}(undef, newly_tethered_buffer_size)

    #package together
    all_data_metal = AllDataMetal(
        positions,
        next_positions,
        effective_radii,
        identifiers,
        tether_points,
        agg_dist_since_grid_sync,
        nascent_to_inserting_ixs,
        nascent_to_substrate_ixs,
        substrate_inserting_ideal_dists,
        coords_to_z_ix,
        z_ix_to_coords,
        agent_cell_z_ixs,
        num_agents_in_cell,
        start_agents_in_cell,
        newly_tethered_agent_ixs
    )

    return all_data_metal
end


"""
    initialise_CPU_metal_data(grid_size::GridSize, params::AllParams)

Initialises CPU-bound data structures necessary for running the simulation on a Metal backend.
"""
function initialise_CPU_metal_data(grid_size::GridSize, params::AllParams)

    #grid_size_metal object (pretty much an exact clone of the CPU version, but GPU safe and without the flags)
    grid_size_metal = GridSizeMetal(
        Float32.(grid_size.dims),
        grid_size.num_cells,
        grid_size.tot_num_cells,
        grid_size.num_agents
    )

    #build the params metal struct (flat, hence passable to GPU)
    params_metal = ParamsMetal(
        Float32(params.system.dt),
        Float32(params.OmpA.radius),
        Float32(params.OmpCF.radius),
        Float32(params.BamA.radius),
        Float32(params.LptD.radius),
        Float32(params.LPS.radius),
        Float32(params.OmpA.tether_radius),
        Float32(params.LptD.tether_radius),
        Float32(params.insertion.mu_attr),
        Float32(params.insertion.mu_rep),
        Float32(params.insertion.k_C),
        Float32(params.force.sensing_radius),
        Float32(params.force.mu_attr_LPS_LPS),
        Float32(params.force.mu_attr_OMP_LPS),
        Float32(params.force.mu_attr_OMP_OMP),
        Float32(params.force.mu_rep),
        Float32(params.force.max_repulsion),
        Float32(params.force.rho),
        Float32(params.force.k_C),
        Float32(params.force.eta)
    )

    return (
        grid_size_metal,
        params_metal
    )

end