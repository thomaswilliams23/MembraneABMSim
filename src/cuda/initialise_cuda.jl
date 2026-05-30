"""
    initialise_system_CUDA(params::AllParams; clear_existing_output::Bool=false, suppress_prints::Bool=false)

Initialises the simulation system using the CUDA GPU backend. Includes additional data 
structures necessary for running the simulation with a GPU.
"""
function initialise_system_CUDA(params::AllParams; clear_existing_output::Bool=false, suppress_prints::Bool=false)

    #build the kernels
    non_force_position_kernel = _non_force_position_kernel_CUDA!(CUDABackend())
    force_kernel = _force_kernel_CUDA!(CUDABackend())

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
            all_data_CUDA,
            grid_size_CUDA,
            params_CUDA
        ) = initialise_system_random_CUDA(force_kernel, params; suppress_prints=suppress_prints)
    else
        #TODO: implement other initialisation methods?
        error("Initialisation type $(params.init.method) not recognised for this device.")
    end

    # return the initialised model
    return (non_force_position_kernel, force_kernel, agents, grid_size, grid, system_flat_cpu, all_data_CUDA, grid_size_CUDA, params_CUDA)
end


"""
    run_equilibration_CUDA!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, 
        all_data_CUDA::AllDataCUDA, grid_size_CUDA::GridSizeCUDA, force_kernel, params::AllParams, params_CUDA::ParamsCUDA, 
        equilibration_time::Float64; suppress_prints::Bool=false)

Runs equilibration for a specified time using the CUDA GPU backend. Synchronises data between CPU and GPU as needed to update the grid and resolve forces.
"""
function run_equilibration_CUDA!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, 
    all_data_CUDA::AllDataCUDA, grid_size_CUDA::GridSizeCUDA, force_kernel, params::AllParams, params_CUDA::ParamsCUDA, 
    equilibration_time::Float64; suppress_prints::Bool=false)

    #run equilibration
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #TODO: make adaptive? calibrate?
    t = 0.0
    time_err = 0.1 * params.system.dt
    while t < equilibration_time - time_err

        if steps_since_grid_sync >= MAX_STEPS_BETWEEN_GRID_SYNC

            copy_data_to_cpu_from_CUDA!(system_flat_cpu, agents, all_data_CUDA, grid_size_CUDA)

            rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
            compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
            put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

            copy_data_to_CUDA!(all_data_CUDA, grid_size_CUDA, system_flat_cpu, grid_size, grid)

            steps_since_grid_sync = 0
        else
            steps_since_grid_sync += 1
        end

        resolve_forces_CUDA!(
            force_kernel, 
            all_data_CUDA, 
            grid_size_CUDA, 
            params_CUDA
        )
        KernelAbstractions.synchronize(CUDABackend())
        copyto!(all_data_CUDA.positions, all_data_CUDA.next_positions[1:2*grid_size_CUDA.num_agents])
        
        t += params.system.dt

        
        #report time
        if abs(t/params.system.vis_dt - round(t/params.system.vis_dt))<time_err
            if !suppress_prints
                @printf "Running equilibration: τ=%5.2f\r" t
            end
        end
    end

    #pass data back to CPU at end of equilibration (if we haven't already during the loop) so that positions and grid are up to date for any subsequent steps
    copy_data_to_cpu_from_CUDA!(system_flat_cpu, agents, all_data_CUDA, grid_size_CUDA)

    if !suppress_prints
        println("\nEquilibration complete.")
    end

end




"""
    run_membrane_shrinkage_CUDA!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, 
        all_data_CUDA::AllDataCUDA, grid_size_CUDA::GridSizeCUDA, force_kernel, params::AllParams, params_CUDA::ParamsCUDA; 
        suppress_prints::Bool=false)

Runs an iterative membrane shrinkage procedure to remove holes in the initial configuration. Shrinks the domain iteratively, equilibrating at each step, until holes are removed 
or a maximum number of shrinkage rounds is reached.
"""
function run_membrane_shrinkage_CUDA!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat_cpu::AllAgentsFlat, 
    all_data_CUDA::AllDataCUDA, grid_size_CUDA::GridSizeCUDA, force_kernel, params::AllParams, params_CUDA::ParamsCUDA; 
    suppress_prints::Bool=false)

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
            copy_data_to_CUDA!(all_data_CUDA, grid_size_CUDA, system_flat_cpu, grid_size, grid) #<- this copies the updated positions and dimensions to the GPU for equilibration

            #equilibrate again
            run_equilibration_CUDA!(agents, grid_size, grid, system_flat_cpu, all_data_CUDA, grid_size_CUDA, force_kernel, params, params_CUDA, shrinkage_equilibration_time; suppress_prints=true)

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
    initialise_system_random_CUDA(force_kernel, params::AllParams)

Initialises agents randomly within the domain and runs equilibration using CUDA GPU backend.
"""
function initialise_system_random_CUDA(force_kernel, params::AllParams; suppress_prints::Bool=false)
    #initialise as per CPU version
    grid_size, grid = initialise_grid(params)
    system_flat_cpu = initialise_flat_cpu(params)
    agents = initialise_agents_random(grid_size, params)

    #initialise CPU-bound CUDA data
    grid_size_CUDA, params_CUDA = initialise_CPU_CUDA_data(grid_size, params)

    #initialise GPU-bound data
    all_data_CUDA = initialise_GPU_CUDA_data(grid_size, params)
    
    #build initial grid
    rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
    compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
    put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)
    grid_size.num_agents_changed = false

    #copy data to CUDA GPU
    copy_data_to_CUDA!(all_data_CUDA, grid_size_CUDA, system_flat_cpu, grid_size, grid)
    copyto!(all_data_CUDA.next_positions, all_data_CUDA.positions[1:2*grid_size_CUDA.num_agents])

    #run equilibration
    if params.init.equilibration_time > 0.0
        run_equilibration_CUDA!(agents, grid_size, grid, system_flat_cpu, all_data_CUDA, grid_size_CUDA, force_kernel, params, params_CUDA, params.init.equilibration_time; suppress_prints=suppress_prints)
    end

    #optionally, shrink the domain to fit the agents after equilibration
    if params.init.shrink_to_size == true
        run_membrane_shrinkage_CUDA!(agents, grid_size, grid, system_flat_cpu, all_data_CUDA, grid_size_CUDA, force_kernel, params, params_CUDA; suppress_prints=suppress_prints)
    end

    #if specified, set all agents as assembled/tethered
    if params.init.complexes_assembled
        assemble_all_agents!(agents, system_flat_cpu)

        #this will also change identifiers and tether points so copy across to GPU
        copyto!(all_data_CUDA.identifiers, system_flat_cpu.identifiers[1:grid_size.num_agents])
        copyto!(all_data_CUDA.tether_points, system_flat_cpu.tether_points[1:2*grid_size.num_agents])
    end

    return (agents, grid_size, grid, system_flat_cpu, all_data_CUDA, grid_size_CUDA, params_CUDA)
end




"""
    initialise_GPU_CUDA_data(grid_size::GridSize, params::AllParams)

Initialises GPU-bound data structures needed for running the simulation on a CUDA backend.
"""
function initialise_GPU_CUDA_data(grid_size::GridSize, params::AllParams)

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
    positions = CuArray{Float32, 1, CUDA.DeviceMemory}(undef, 2*num_agents_init)
    next_positions = CuArray{Float32, 1, CUDA.DeviceMemory}(undef, 2*num_agents_init)
    effective_radii = CuArray{Float32, 1, CUDA.DeviceMemory}(undef, num_agents_init)
    identifiers = CuArray{Int, 1, CUDA.DeviceMemory}(undef, num_agents_init)
    tether_points = CuArray{Float32, 1, CUDA.DeviceMemory}(undef, 2*num_agents_init)
    agg_dist_since_grid_sync = CuArray{Float32, 1, CUDA.DeviceMemory}(undef, num_agents_init)

    nascent_to_inserting_ixs = CuArray{Int, 1, CUDA.DeviceMemory}(undef, nascent_buffer_size)
    nascent_to_substrate_ixs = CuArray{Int, 1, CUDA.DeviceMemory}(undef, nascent_buffer_size)
    substrate_inserting_ideal_dists = CuArray{Float32, 1, CUDA.DeviceMemory}(undef, nascent_buffer_size)
    
    coords_to_z_ix = CuArray{Int, 1, CUDA.DeviceMemory}(undef, grid_size.tot_num_cells)
    z_ix_to_coords = CuArray{Int, 1, CUDA.DeviceMemory}(undef, 2*grid_size.tot_num_cells)
    agent_cell_z_ixs = CuArray{Int, 1, CUDA.DeviceMemory}(undef, num_agents_init)
    num_agents_in_cell = CuArray{Int, 1, CUDA.DeviceMemory}(undef, grid_size.tot_num_cells)
    start_agents_in_cell = CuArray{Int, 1, CUDA.DeviceMemory}(undef, grid_size.tot_num_cells)

    newly_tethered_agent_ixs = CuArray{Int, 1, CUDA.DeviceMemory}(undef, newly_tethered_buffer_size)

    #package together
    all_data_CUDA = AllDataCUDA(
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

    return all_data_CUDA
end


"""
    initialise_CPU_CUDA_data(grid_size::GridSize, params::AllParams)

Initialises CPU-bound data structures necessary for running the simulation on a CUDA backend.
"""
function initialise_CPU_CUDA_data(grid_size::GridSize, params::AllParams)

    #grid_size_CUDA object (pretty much an exact clone of the CPU version, but GPU safe and without the flags)
    grid_size_CUDA = GridSizeCUDA(
        Float32.(grid_size.dims),
        grid_size.num_cells,
        grid_size.tot_num_cells,
        grid_size.num_agents
    )

    #build the params CUDA struct (flat, hence passable to GPU)
    params_CUDA = ParamsCUDA(
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
        grid_size_CUDA,
        params_CUDA
    )

end
