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
    set_up_output_directory(params.system.output_dir; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints)

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
        error("Initialisation type $(params.init.method) not recognised.")
    end

    # return the initialised model
    return (non_force_position_kernel, force_kernel, agents, grid_size, grid, system_flat_cpu, all_data_metal, grid_size_metal, params_metal)
end




"""
    run_equilibration_metal!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, 
        all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, force_kernel, params::AllParams, params_metal::ParamsMetal, 
        equilibration_time::Float64; suppress_prints::Bool=false)

Runs equilibration for a specified time using the Metal GPU backend. Synchronises data between CPU and GPU as needed to update the grid and resolve forces.
"""
function run_equilibration_metal!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, 
    all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, force_kernel, params::AllParams, params_metal::ParamsMetal, 
    equilibration_time::Float64; suppress_prints::Bool=false)

    #run equilibration
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #TODO: make adaptive? calibrate?
    t = 0.0
    time_err = 0.1 * params.system.dt
    while t < equilibration_time - time_err

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

    #pass data back to CPU at end of equilibration (if we haven't already during the loop) so that positions and grid are up to date for any subsequent steps
    copy_data_to_cpu_from_metal!(system_flat_cpu, agents, all_data_metal, grid_size_metal)

    if !suppress_prints
        println("\nEquilibration complete.")
    end

end




"""
    run_membrane_shrinkage_metal!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, force_kernel, params::AllParams, params_metal::ParamsMetal; suppress_prints::Bool=false)

Runs an iterative membrane shrinkage procedure to remove holes in the initial configuration. Shrinks the domain iteratively, equilibrating at each step, until holes are removed 
or a maximum number of shrinkage rounds is reached.
"""
function run_membrane_shrinkage_metal!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, force_kernel, params::AllParams, params_metal::ParamsMetal; suppress_prints::Bool=false)

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
            shrinkage_scale_factor = (1.0 - shrinkage_factor)
            apply_scale_factor!(agents, system_flat_cpu, grid_size, shrinkage_scale_factor) #<- this handles the CPU-side data
            copy_data_to_metal!(all_data_metal, grid_size_metal, system_flat_cpu, grid_size, grid) #<- this copies the updated positions and dimensions to the GPU for equilibration

            #equilibrate again
            run_equilibration_metal!(agents, grid_size, grid, system_flat_cpu, all_data_metal, grid_size_metal, force_kernel, params, params_metal, shrinkage_equilibration_time; suppress_prints=true)

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
    copyto!(all_data_metal.next_positions, all_data_metal.positions[1:2*grid_size_metal.num_agents])

    #run equilibration
    if params.init.equilibration_time > 0.0
        run_equilibration_metal!(agents, grid_size, grid, system_flat_cpu, all_data_metal, grid_size_metal, force_kernel, params, params_metal, params.init.equilibration_time; suppress_prints=suppress_prints)
    end

    #optionally, shrink the domain to fit the agents after equilibration
    if params.init.shrink_to_size == true
        run_membrane_shrinkage_metal!(agents, grid_size, grid, system_flat_cpu, all_data_metal, grid_size_metal, force_kernel, params, params_metal; suppress_prints=suppress_prints)
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