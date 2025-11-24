"""
    initialise_system_metal(params::AllParams)

Initialises the simulation system using the Metal GPU backend. Includes additional data 
structures necessary for running the simulation with a GPU.
"""
function initialise_system_metal(params::AllParams)

    #build the kernels
    non_force_position_kernel = _non_force_position_kernel!(MetalBackend())
    force_kernel = _force_kernel!(MetalBackend())

    # set up output directory structure
    set_up_output_directory(params.system.output_dir)

    # decide which initialisation to use
    if params.init.method == "random"
        println("Initialising model with random distribution of agents...")
        (
            agents, 
            grid_size,
            grid,
            system_flat_cpu,
            all_data_metal,
            grid_size_metal,
            params_metal
        ) = initialise_system_random_metal(force_kernel, params)
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
function initialise_system_random_metal(force_kernel, params::AllParams)

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

            copy_data_to_cpu!(system_flat_cpu, agents, all_data_metal, grid_size_metal)

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
            @printf "Running equilibration: τ=%5.2f\r" t
        end
    end

    println("\nEquilibration complete.")

    #now copy across to cpu
    if params.init.equilibration_time > time_err
        copy_data_to_cpu!(system_flat_cpu, agents, all_data_metal, grid_size_metal)
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


"""
    copy_data_to_metal!(all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, system_flat_cpu::AllAgentsFlat, grid_size::GridSize, grid::SimGrid)

Copies agent and grid data from CPU to GPU, resizing buffers if needed.
"""
function copy_data_to_metal!(all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, system_flat_cpu::AllAgentsFlat, grid_size::GridSize, grid::SimGrid)

    #handle resizing
    curr_agent_vec_capacity = length(all_data_metal.effective_radii)
    curr_grid_vec_capacity = length(all_data_metal.num_agents_in_cell)
    curr_nascent_capacity = length(all_data_metal.nascent_to_substrate_ixs)
    if grid_size.num_agents>curr_agent_vec_capacity
        size_increase_ratio = 1.25
        new_agent_vec_size = ceil(Int, size_increase_ratio*grid_size.num_agents)
        resize!(all_data_metal.positions, 2*new_agent_vec_size)
        resize!(all_data_metal.next_positions, 2*new_agent_vec_size)
        resize!(all_data_metal.effective_radii, new_agent_vec_size)
        resize!(all_data_metal.identifiers, new_agent_vec_size)
        resize!(all_data_metal.tether_points, 2*new_agent_vec_size)
        resize!(all_data_metal.agg_dist_since_grid_sync, new_agent_vec_size)
        resize!(all_data_metal.agent_cell_z_ixs, new_agent_vec_size)
    end
    if grid_size.tot_num_cells>curr_grid_vec_capacity
        size_increase_ratio = 1.25
        new_grid_vec_size = ceil(Int, size_increase_ratio*grid_size.tot_num_cells)
        resize!(all_data_metal.coords_to_z_ix, new_grid_vec_size)
        resize!(all_data_metal.z_ix_to_coords, 2*new_grid_vec_size)
        resize!(all_data_metal.num_agents_in_cell, new_grid_vec_size)
        resize!(all_data_metal.start_agents_in_cell, new_grid_vec_size)
    end
    if length(system_flat_cpu.nascent_to_substrate_ixs) > curr_nascent_capacity
        size_increase_ratio = 1.25
        new_vec_size = ceil(Int, size_increase_ratio * length(system_flat_cpu.nascent_to_substrate_ixs))
        resize!(all_data_metal.nascent_to_substrate_ixs, new_vec_size)
        resize!(all_data_metal.nascent_to_inserting_ixs, new_vec_size)
        resize!(all_data_metal.substrate_inserting_ideal_dists, new_vec_size)
    end


    #copy across flat system data
    copyto!(all_data_metal.positions, system_flat_cpu.positions[1:2*grid_size.num_agents])
    copyto!(all_data_metal.next_positions, system_flat_cpu.positions[1:2*grid_size.num_agents])
    copyto!(all_data_metal.effective_radii, system_flat_cpu.effective_radii[1:grid_size.num_agents])
    copyto!(all_data_metal.identifiers, system_flat_cpu.identifiers[1:grid_size.num_agents])
    copyto!(all_data_metal.tether_points, system_flat_cpu.tether_points[1:2*grid_size.num_agents])
    copyto!(all_data_metal.agg_dist_since_grid_sync, system_flat_cpu.agg_dist_since_grid_sync[1:grid_size.num_agents])

    copyto!(all_data_metal.nascent_to_inserting_ixs, system_flat_cpu.nascent_to_inserting_ixs)
    copyto!(all_data_metal.nascent_to_substrate_ixs, system_flat_cpu.nascent_to_substrate_ixs)
    copyto!(all_data_metal.substrate_inserting_ideal_dists, system_flat_cpu.substrate_inserting_ideal_dists)

    #copy across grid data
    copyto!(all_data_metal.agent_cell_z_ixs, grid.agent_cell_z_ixs[1:grid_size.num_agents])

    copyto!(all_data_metal.num_agents_in_cell, grid.num_agents_in_cell[1:grid_size.tot_num_cells])
    copyto!(all_data_metal.start_agents_in_cell, grid.start_agents_in_cell[1:grid_size.tot_num_cells])

    #update grid metal data (CPU-bound)
    grid_size_metal.dims = Float32.(grid_size.dims)
    grid_size_metal.num_agents = grid_size.num_agents

    #morton lookup tables and grid size only need updating if grid has been resized
    if grid_size.grid_size_changed
        copyto!(all_data_metal.coords_to_z_ix, grid.coords_to_z_ix[1:grid_size.tot_num_cells])
        copyto!(all_data_metal.z_ix_to_coords, grid.z_ix_to_coords[1:2*grid_size.tot_num_cells])
        grid_size_metal.num_cells = grid_size.num_cells
        grid_size_metal.tot_num_cells = grid_size.tot_num_cells
    end

end




"""
    copy_data_to_cpu!(system_flat_cpu::AllAgentsFlat, agents::AllAgents, all_data_metal::AllDataMetal)

Copies position and aggregate distance data from GPU to CPU.
"""
function copy_data_to_cpu!(system_flat_cpu::AllAgentsFlat, agents::AllAgents, all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal)

    #make sure all previous GPU operations are complete
    KernelAbstractions.synchronize(MetalBackend())

    #copy the new positions and aggregate distances out to the cpu
    copyto!(system_flat_cpu.positions, all_data_metal.next_positions[1:2*grid_size_metal.num_agents])
    copyto!(system_flat_cpu.agg_dist_since_grid_sync, all_data_metal.agg_dist_since_grid_sync[1:grid_size_metal.num_agents])
    copyto!(system_flat_cpu.tether_points, all_data_metal.tether_points[1:2*grid_size_metal.num_agents])
    
    #write new positions into AllAgents structure
    all_agents = Iterators.flatten((agents.OMP.OmpA, agents.OMP.OmpCF, agents.OMP.LptD, agents.OMP.BamA, 
                                    agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
    for agent in all_agents
        agent_ix = agent.index
        if agent_ix>grid_size_metal.num_agents
            continue
        end
        sorted_ix = system_flat_cpu.agent_ix_to_sorted_ix[agent_ix]
        agent.position = SVector{2, Float64}(system_flat_cpu.positions[2*sorted_ix-1], system_flat_cpu.positions[2*sorted_ix])
    end

    #now iterate through all tethered agents and update their tether points
    all_tetherable_agents = Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD))
    for agent in all_tetherable_agents
        if agent.is_tethered
            agent_ix = agent.index
            if agent_ix>grid_size_metal.num_agents
                continue
            end
            sorted_ix = system_flat_cpu.agent_ix_to_sorted_ix[agent_ix]
            agent.tether_point = SVector{2, Float64}(
                system_flat_cpu.tether_points[2*sorted_ix-1], 
                system_flat_cpu.tether_points[2*sorted_ix]
            )
        end
    end
end



"""
    copy_data_to_cpu_yn(agents::AllAgents, t::Float64)

Decides whether to copy data back to CPU this time step based on agent types and current time.
"""
function copy_data_to_cpu_yn(BamA_agents::Vector{BamAAgent}, params::AllParams, t::Float64)

    t_next_time_step = t + params.system.dt

    #first check if there is a BamA in the bound state, and we will attempt insertion on the next time step
    attempt_dt_err = 0.1 * params.system.dt
    if abs(t_next_time_step/params.insertion.attempt_dt - round(t_next_time_step/params.insertion.attempt_dt))<attempt_dt_err
        for BamA in BamA_agents
            if BamA.insertion_state == "bound"
                return true
            end
        end
    end

    #next check if we need to copy data out for visualisation this time step
    time_err = 0.1 * params.system.dt
    if abs(t_next_time_step/params.system.vis_dt - round(t_next_time_step/params.system.vis_dt))<time_err
        return true
    end

    return false

end




"""
    update_nascent_promotion_metal!(all_data_metal::AllDataMetal, system_flat_cpu::AllAgentsFlat, grid_size::GridSize)

In the case where a nascent agent has been promoted, copy across all updated nascent-related data to the GPU.
"""
function update_nascent_promotion_metal!(all_data_metal::AllDataMetal, system_flat_cpu::AllAgentsFlat, grid_size::GridSize)

    #check that no GPU processes are ongoing
    KernelAbstractions.synchronize(MetalBackend())

    #update nascent to inserting indices
    copyto!(all_data_metal.nascent_to_inserting_ixs, system_flat_cpu.nascent_to_inserting_ixs)
    copyto!(all_data_metal.nascent_to_substrate_ixs, system_flat_cpu.nascent_to_substrate_ixs)
    copyto!(all_data_metal.substrate_inserting_ideal_dists, system_flat_cpu.substrate_inserting_ideal_dists)

    #update identifiers
    copyto!(all_data_metal.identifiers, system_flat_cpu.identifiers[1:grid_size.num_agents])

    #reset nascent_promoted flag
    grid_size.nascent_promoted = false
end


"""
    update_newly_tethered_agent_ixs_metal!(all_data_metal::AllDataMetal, newly_tethered_agent_ixs::Vector{Int})

In the case where agents have just become tethered, copy across their indices to the GPU.
"""
function update_newly_tethered_agent_ixs_metal!(all_data_metal::AllDataMetal, newly_tethered_agent_ixs::Vector{Int})

    #check that no GPU processes are ongoing
    KernelAbstractions.synchronize(MetalBackend())

    #handle resizing
    curr_capacity = length(all_data_metal.newly_tethered_agent_ixs)
    if length(newly_tethered_agent_ixs)>curr_capacity
        size_increase_ratio = 1.25
        new_vec_size = ceil(Int, size_increase_ratio*length(newly_tethered_agent_ixs))
        resize!(all_data_metal.newly_tethered_agent_ixs, new_vec_size)
    end

    #copy across
    copyto!(all_data_metal.newly_tethered_agent_ixs, newly_tethered_agent_ixs)

end



"""
    resolve_forces_metal!(force_kernel, all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, params_metal::ParamsMetal)

Main function for computing forces and updating agent positions on the Metal GPU.
"""
function resolve_forces_metal!(force_kernel, all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal, params_metal::ParamsMetal)

    #ensure all previous GPU operations are complete
    KernelAbstractions.synchronize(MetalBackend())

    #update positions field on GPU
    copyto!(all_data_metal.positions, all_data_metal.next_positions[1:2*grid_size_metal.num_agents])

    #some calculations needed for kernel
    max_effective_radius = Float32(max(params_metal.OmpA_radius, params_metal.OmpCF_radius, params_metal.LptD_radius, params_metal.BamA_radius, params_metal.LPS_radius))
    max_agg_dist = maximum(all_data_metal.agg_dist_since_grid_sync)

    #run the force kernel
    force_kernel(
        all_data_metal.positions,
        all_data_metal.next_positions,
        all_data_metal.effective_radii,
        all_data_metal.identifiers,
        all_data_metal.tether_points,
        all_data_metal.agg_dist_since_grid_sync,
        all_data_metal.nascent_to_inserting_ixs,
        all_data_metal.nascent_to_substrate_ixs,
        all_data_metal.substrate_inserting_ideal_dists,
        all_data_metal.coords_to_z_ix,
        all_data_metal.z_ix_to_coords,
        all_data_metal.agent_cell_z_ixs,
        all_data_metal.num_agents_in_cell,
        all_data_metal.start_agents_in_cell,
        grid_size_metal.dims,
        grid_size_metal.num_cells,
        max_effective_radius,
        max_agg_dist,
        params_metal;
        ndrange=grid_size_metal.num_agents
    )
end


"""
    @kernel function compute_next_positions_metal!(
        positions::MtlDeviceVector{Float32},
        next_positions::MtlDeviceVector{Float32},
        effective_radii::MtlDeviceVector{Float32},
        identifiers::MtlDeviceVector{Int},
        tether_points::MtlDeviceVector{Float32},
        agg_dist_since_grid_sync::MtlDeviceVector{Float32},
        nascent_to_inserting_ixs::MtlDeviceVector{Int},
        nascent_to_substrate_ixs::MtlDeviceVector{Int},
        substrate_inserting_ideal_dists::MtlDeviceVector{Float32},
        coords_to_z_ix::MtlDeviceVector{Int},
        z_ix_to_coords::MtlDeviceVector{Int},
        agent_cell_z_ixs::MtlDeviceVector{Int},
        num_agents_in_cell::MtlDeviceVector{Int},
        start_agents_in_cell::MtlDeviceVector{Int},
        dims::SVector{2, Float32},
        num_cells::SVector{2, Int},
        max_effective_radius::Float32,
        max_agg_dist::Float32,
        params_metal::ParamsMetal
    )

Metal GPU kernel for computing next agent positions based on forces.
"""
@kernel function _force_kernel!(
    positions::MtlDeviceVector{Float32},
    next_positions::MtlDeviceVector{Float32},
    effective_radii::MtlDeviceVector{Float32},
    identifiers::MtlDeviceVector{Int},
    tether_points::MtlDeviceVector{Float32},
    agg_dist_since_grid_sync::MtlDeviceVector{Float32},
    nascent_to_inserting_ixs::MtlDeviceVector{Int},
    nascent_to_substrate_ixs::MtlDeviceVector{Int},
    substrate_inserting_ideal_dists::MtlDeviceVector{Float32},
    coords_to_z_ix::MtlDeviceVector{Int},
    z_ix_to_coords::MtlDeviceVector{Int},
    agent_cell_z_ixs::MtlDeviceVector{Int},
    num_agents_in_cell::MtlDeviceVector{Int},
    start_agents_in_cell::MtlDeviceVector{Int},
    dims::SVector{2, Float32},
    num_cells::SVector{2, Int},
    max_effective_radius::Float32,
    max_agg_dist::Float32,
    params_metal::ParamsMetal
    )

    # Get sorted index
    sorted_ix = @index(Global)

    #parse this agent's identifier
    identifier = identifiers[sorted_ix]
    is_tethered, agent_type_num, is_nascent, is_inserting, nascent_ix = parse_identifier_metal(identifier)

    #if inserting or nascent, get the substrate/inserting ix so we can ignore it in attraction/repulsion force calculations
    if is_inserting==1
        sorted_substrate_inserting_ix = nascent_to_substrate_ixs[nascent_ix]
    elseif is_nascent==1
        sorted_substrate_inserting_ix = nascent_to_inserting_ixs[nascent_ix]
    else
        sorted_substrate_inserting_ix = -1
    end

    #tally forces acting on this agent (ignore substrate/inserting agent, if one exists)
    resultant_force = tally_attr_rep_forces_metal(
        sorted_ix,
        sorted_substrate_inserting_ix,
        positions,
        effective_radii,
        identifiers,
        agg_dist_since_grid_sync,
        coords_to_z_ix,
        z_ix_to_coords,
        agent_cell_z_ixs,
        num_agents_in_cell,
        start_agents_in_cell,
        dims,
        num_cells,
        max_effective_radius,
        max_agg_dist,
        params_metal
    )

    #if this agent has a substrate/inserting agent, compute the spring force between them
    if sorted_substrate_inserting_ix>0
        agent_pos = SVector{2, Float32}(
            positions[sorted_ix*2-1], 
            positions[sorted_ix*2]
        )
        neigh_pos = SVector{2, Float32}(
            positions[sorted_substrate_inserting_ix*2-1], 
            positions[sorted_substrate_inserting_ix*2]
        )
        resultant_force += compute_inserting_substrate_force_metal(
            agent_pos,
            neigh_pos,
            dims,
            substrate_inserting_ideal_dists[nascent_ix],
            params_metal.insertion_mu_attr,
            params_metal.insertion_mu_rep,
            params_metal.max_repulsion,
            params_metal.rho,
            params_metal.insertion_k_C
        )
    end

    #get actual radius
    if is_nascent == 1 && agent_type_num != 2  #ie. is a nascent OMP
        if agent_type_num == 1 #OmpA
            act_rad = params_metal.OmpA_radius
        elseif agent_type_num == 3 #OmpCF
            act_rad = params_metal.OmpCF_radius
        elseif agent_type_num == 5 #BamA
            act_rad = params_metal.BamA_radius
        elseif agent_type_num == 7 #LptD
            act_rad = params_metal.LptD_radius
        else
            act_rad = 0.0f0  #should never happen
        end
    else
        #otherwise, actual radius is just effective radius
        act_rad = effective_radii[sorted_ix]
    end

    #compute the displacement from the resultant force
    displacement = (params_metal.dt/(params_metal.eta*act_rad))*resultant_force

    #compute proposal position from force sum
    agent_pos = SVector{2, Float32}(positions[2*sorted_ix-1], positions[2*sorted_ix])
    next_pos = agent_pos + displacement
    next_pos = SVector{2, Float32}(
        mod(next_pos[1], dims[1]),
        mod(next_pos[2], dims[2])
    )

    #check if agent has a tether and if proposal position exceeds tether length
    if is_tethered==1
        tether_pos = SVector{2, Float32}(tether_points[2*sorted_ix-1], tether_points[2*sorted_ix])
        if agent_type_num == 1 #OmpA
            tether_length = params_metal.OmpA_tether_radius
        elseif agent_type_num == 7 #LptD
            tether_length = params_metal.LptD_tether_radius
        else
            tether_length = 0.0f0  #should never happen
        end
        if shortest_distance_metal(next_pos, tether_pos, dims)>tether_length
            #make next position same as old position
            next_pos = SVector{2, Float32}(positions[2*sorted_ix-1], positions[2*sorted_ix])
        else
            #if proposal position is valid, update aggregated distance
            norm_displacement = sqrt(displacement[1]^2 + displacement[2]^2)
            agg_dist_since_grid_sync[sorted_ix] += norm_displacement
        end
    else
        #if no tether, proposal automatically accepted, update aggregated distance
        norm_displacement = sqrt(displacement[1]^2 + displacement[2]^2)
        agg_dist_since_grid_sync[sorted_ix] += norm_displacement
    end

    #write to the new positions vector
    next_positions[2*sorted_ix-1] = next_pos[1]
    next_positions[2*sorted_ix] = next_pos[2]

end



"""
    function tally_attr_rep_forces_metal(
        sorted_ix::Int, 
        sorted_substrate_inserting_ix::Int,
        positions::MtlDeviceVector{Float32},
        effective_radii::MtlDeviceVector{Float32},
        identifiers::MtlDeviceVector{Int},
        agg_dist_since_grid_sync::MtlDeviceVector{Float32},
        coords_to_z_ix::MtlDeviceVector{Int},
        z_ix_to_coords::MtlDeviceVector{Int},
        agent_cell_z_ixs::MtlDeviceVector{Int},
        num_agents_in_cell::MtlDeviceVector{Int},
        start_agents_in_cell::MtlDeviceVector{Int},
        dims::SVector{2, Float32},
        num_cells::SVector{2, Int},
        max_effective_radius::Float32,
        max_agg_dist::Float32,
        params_metal::ParamsMetal
    )

Tally the total attraction/repulsion forces acting on an agent from its neighbours. Directly analogous to the CPU version.
"""
@inline function tally_attr_rep_forces_metal(
        sorted_ix::Int, 
        sorted_substrate_inserting_ix::Int,
        positions::MtlDeviceVector{Float32},
        effective_radii::MtlDeviceVector{Float32},
        identifiers::MtlDeviceVector{Int},
        agg_dist_since_grid_sync::MtlDeviceVector{Float32},
        coords_to_z_ix::MtlDeviceVector{Int},
        z_ix_to_coords::MtlDeviceVector{Int},
        agent_cell_z_ixs::MtlDeviceVector{Int},
        num_agents_in_cell::MtlDeviceVector{Int},
        start_agents_in_cell::MtlDeviceVector{Int},
        dims::SVector{2, Float32},
        num_cells::SVector{2, Int},
        max_effective_radius::Float32,
        max_agg_dist::Float32,
        params_metal::ParamsMetal
    ) :: SVector{2, Float32}

    #initialise
    resultant_force = SVector{2, Float32}(0.0f0, 0.0f0)

    #determine the cell the agent is in
    agent_cell_z_ix = agent_cell_z_ixs[sorted_ix]
    agent_cell_ix = z_ix_to_coords[2*agent_cell_z_ix-1]
    agent_cell_jx = z_ix_to_coords[2*agent_cell_z_ix]

    #calculate how wide around this cell we need to search for neighbours
    agent_buffer_radius = effective_radii[sorted_ix] + params_metal.sensing_radius + max_effective_radius
    
    #add in the error from movement of this agent and all possible neighbours since last grid sync
    agent_buffer_radius += max_agg_dist + agg_dist_since_grid_sync[sorted_ix]
    
    #determine how many cells to search in each direction
    cell_buffer_num_x = fast_floor_int32(agent_buffer_radius/(dims[1]/num_cells[1])) + 1
    cell_buffer_num_y = fast_floor_int32(agent_buffer_radius/(dims[2]/num_cells[2])) + 1
    
    #cap cell intervals to avoid double-searching
    cell_buffer_num_x = min(cell_buffer_num_x, div(num_cells[1], 2)+1)
    cell_buffer_num_y = min(cell_buffer_num_y, div(num_cells[2], 2)+1)

    #loop over Moore neighbourhood
    x_shift = -cell_buffer_num_x
    while x_shift <= cell_buffer_num_x
        y_shift = -cell_buffer_num_y
        while y_shift <= cell_buffer_num_y

            #account for periodic boundaries
            neigh_cell_x = mod(agent_cell_ix + x_shift - 1, num_cells[1]) + 1
            neigh_cell_y = mod(agent_cell_jx + y_shift - 1, num_cells[2]) + 1
            neigh_cell_z_ix = coords_to_z_ix[(neigh_cell_y-1)*num_cells[1] + neigh_cell_x]

            #loop over agents in cell if not empty
            start_agent = start_agents_in_cell[neigh_cell_z_ix]
            num_agents = num_agents_in_cell[neigh_cell_z_ix]
            
            for sorted_n_ix = start_agent:(start_agent + num_agents - 1)

                #avoid self-interaction and attr-rep forces between substrate-inserting pairs
                if sorted_n_ix!=sorted_ix && sorted_n_ix!=sorted_substrate_inserting_ix

                    #work out which type of interaction (for determining mu_attr)
                    num_OMPs_in_interaction = 
                        (identifiers[sorted_ix] & 1) + 
                        (identifiers[sorted_n_ix] & 1)
                    if num_OMPs_in_interaction==0
                        mu_attr = params_metal.mu_attr_LPS_LPS
                    elseif num_OMPs_in_interaction==1
                        mu_attr = params_metal.mu_attr_OMP_LPS
                    elseif num_OMPs_in_interaction==2
                        mu_attr = params_metal.mu_attr_OMP_OMP
                    else
                        mu_attr = 0.0f0  #should never happen
                    end

                    #pull out agent position
                    agent_position = SVector{2, Float32}(positions[2*sorted_ix-1], positions[2*sorted_ix])
                    neighbour_position = SVector{2, Float32}(positions[2*sorted_n_ix-1], positions[2*sorted_n_ix])

                    #compute force between agent and neighbour
                    resultant_force += compute_attr_rep_force_metal(
                        agent_position,
                        effective_radii[sorted_ix],
                        neighbour_position,
                        effective_radii[sorted_n_ix],
                        dims,
                        params_metal.sensing_radius,
                        mu_attr,
                        params_metal.mu_rep,
                        params_metal.max_repulsion,
                        params_metal.rho,
                        params_metal.k_C
                    )
            
                end
            end
            y_shift += 1
        end
        x_shift += 1
    end

    return resultant_force
end



"""
    function compute_attr_rep_force_metal(agent_pos::SVector{2, Float32}, agent_rad::Float32, 
                                          neighbour_pos::SVector{2, Float32}, neighbour_rad::Float32, 
                                          dims::SVector{2, Float32}, sensing_radius::Float32, 
                                          mu_attr::Float32, mu_rep::Float32, max_repulsion::Float32,
                                          rho::Float32, k_C::Float32)
    
Compute the attraction/repulsion force between two agents in the metal simulation.
"""
@inline function compute_attr_rep_force_metal(agent_pos::SVector{2, Float32}, agent_rad::Float32, 
                                      neighbour_pos::SVector{2, Float32}, neighbour_rad::Float32, 
                                      dims::SVector{2, Float32}, sensing_radius::Float32, 
                                      mu_attr::Float32, mu_rep::Float32, max_repulsion::Float32,
                                      rho::Float32, k_C::Float32) :: SVector{2, Float32}

    #compute vector and distance between agents
    force_vec = shortest_vec_metal(agent_pos, neighbour_pos, dims)
    dist = sqrt(force_vec[1]*force_vec[1] + force_vec[2]*force_vec[2])

    #catch the case where distance is extremely small (avoid div by zero)
    dist_eps = Float32(1e-8)
    if dist<dist_eps
        return SVector{2, Float32}(0.0, 0.0)
    end

    #check if within sensing radius
    if dist > agent_rad + neighbour_rad + sensing_radius
        force = SVector{2, Float32}(0.0f0, 0.0f0)
    else

        ideal_dist = agent_rad + neighbour_rad

        #repulsion
        force_mag=0
        if dist<=rho*ideal_dist
            force_mag = max_repulsion
        elseif dist<ideal_dist && dist>rho*ideal_dist
            force_mag = mu_rep*ideal_dist*log((dist-rho*ideal_dist)/(1.0f0-rho)*ideal_dist)
            if force_mag < max_repulsion
                force_mag = max_repulsion
            end
        #attraction
        else
            norm_dist = (dist - ideal_dist)/((1.0f0-rho)*ideal_dist)
            force_mag = mu_attr*ideal_dist*norm_dist*exp(-k_C*norm_dist)
        end
        force = (force_mag/dist)*force_vec
    end

    return force
end


"""
    compute_inserting_substrate_force_metal(agent_pos::SVector{2, Float32}, neighbour_pos::SVector{2, Float32},
                                           dims::SVector{2, Float32}, ideal_dist::Float32, 
                                           mu_attr::Float32, mu_rep::Float32, k_C::Float32)

Compute the force between an inserting agent and the substrate agent it is inserting.
"""
@inline function compute_inserting_substrate_force_metal(agent_pos::SVector{2, Float32}, neighbour_pos::SVector{2, Float32},
                                           dims::SVector{2, Float32}, ideal_dist::Float32, 
                                           mu_attr::Float32, mu_rep::Float32, max_repulsion::Float32,
                                           rho::Float32, k_C::Float32) :: SVector{2, Float32}

    #handle case where ideal distance is zero or very small
    ideal_dist_eps = Float32(1e-8)
    if ideal_dist<ideal_dist_eps
        return SVector{2, Float32}(0.0f0, 0.0f0)
    end
    
    #otherwise, compute vector and distance between agents
    force_vec = shortest_vec_metal(agent_pos, neighbour_pos, dims)
    dist = sqrt(force_vec[1]*force_vec[1] + force_vec[2]*force_vec[2])

    #catch the case where the actual distance is extremely small (avoid div by zero)
    dist_eps = Float32(1e-8)
    if dist<dist_eps
        return SVector{2, Float32}(0.0, 0.0)
    end

    #repulsion
    force_mag=0.0f0
    if dist<=rho*ideal_dist
        force_mag = max_repulsion
    elseif dist<ideal_dist && dist>rho*ideal_dist
        force_mag = mu_rep*ideal_dist*log((dist-rho*ideal_dist)/(1.0f0-rho)*ideal_dist)
        if force_mag < max_repulsion
            force_mag = max_repulsion
        end
    #attraction
    else
        norm_dist = (dist - ideal_dist)/((1.0f0-rho)*ideal_dist)
        force_mag = mu_attr*ideal_dist*norm_dist*exp(-k_C*norm_dist)
    end
    force = (force_mag/dist)*force_vec
    
    return force

end


"""
    shortest_vec_metal(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: SVector{2, Float32}

Returns the shortest vector between two positions, accounting for periodic boundaries.
"""
@inline function shortest_vec_metal(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: SVector{2, Float32}
    raw_vec = pos2 - pos1
    # Replace broadcast operations with component-wise
    wrapped_vec = SVector{2, Float32}(
        raw_vec[1] - round(raw_vec[1] / dims[1]) * dims[1],
        raw_vec[2] - round(raw_vec[2] / dims[2]) * dims[2]
    )
    return wrapped_vec
end

"""
    shortest_distance_metal(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: Float32

Returns the shortest distance between two positions, accounting for periodic boundaries.
"""
@inline function shortest_distance_metal(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: Float32
    vec = shortest_vec_metal(pos1, pos2, dims)
    return sqrt(vec[1]*vec[1] + vec[2]*vec[2])
end


"""
    fast_floor_int32(x::Float32):: Int32

Efficiently computes the floor of a Float32 value and returns it as Int32. GPU-safe.
"""
@inline function fast_floor_int32(x::Float32):: Int32
    i = unsafe_trunc(Int32, x)        # GPU-safe, direct LLVM fptosi
    i -= (x < Float32(i))             # subtract 1 if x < i (emulates floor)
    return i
end



"""
    parse_identifier_metal(identifier::Int)

Parses an agent identifier integer into its constituent properties for Metal GPU kernels.
Returns a SVector of agent properties.
"""
@inline function parse_identifier_metal(identifier::Int)

    is_tethered = identifier < 0
    if is_tethered
        identifier *= -1
    end
    is_nascent = identifier > 1000
    if is_nascent
        is_inserting = false
        identifier -= 1000
    else
        is_inserting = identifier > 10
    end
    nascent_ix = fast_floor_int32(identifier / 10.0f0)
    agent_type_num = identifier - nascent_ix * 10

    agent_data = SVector{5, Int}(is_tethered, agent_type_num, is_nascent, is_inserting, nascent_ix)

    return agent_data
end


"""
    compute_non_force_position_changes!(non_force_position_kernel, agents::AllAgents, all_data_metal::AllDataMetal,
                                       grid_size::GridSize, grid_size_metal::GridSizeMetal,
                                       effective_rad_incs_metal::SVector{7, Float32},
                                       ideal_dist_incs_metal::SVector{7, Float32},
                                       nascent_added_area_lookup::NascentAddedAreaLookup,
                                       newly_tethered_agent_ixs::Vector{Int},
                                       params::AllParams, t::Float64)

Applies non-force-driven changes to agent positions using a Metal GPU kernel. Specifically, this handles:
 - rescaling of positions due to domain size change
 - formation of new tethers
 - updating nascent-inserting ideal distances
 - updating effective radii of nascent agents
"""
function compute_non_force_position_changes!(
    non_force_position_kernel,
    agents::AllAgents,
    all_data_metal::AllDataMetal,
    grid_size::GridSize,
    grid_size_metal::GridSizeMetal,
    effective_rad_incs_metal::SVector{7, Float32},
    ideal_dist_incs_metal::SVector{7, Float32},
    nascent_added_area_lookup::NascentAddedAreaLookup,
    newly_tethered_agent_ixs::Vector{Int},
    params::AllParams,
    t::Float64
    )


    #make sure all previous GPU operations are complete
    KernelAbstractions.synchronize(MetalBackend())

    #update positions field on GPU
    copyto!(all_data_metal.positions, all_data_metal.next_positions[1:2*grid_size_metal.num_agents])


    #first, compute the added area
    added_area_this_timestep = compute_added_area(agents, params, nascent_added_area_lookup, t)
    
    #early exit
    added_area_err = 1e-10
    if added_area_this_timestep<added_area_err && length(newly_tethered_agent_ixs)==0
        return
    end

    #compute the rescaling factor
    prev_area = prod(grid_size.dims)
    scaled_added_area = added_area_this_timestep/params.system.density
    scale_factor = sqrt((prev_area + scaled_added_area)/prev_area)
    scale_factor_metal = Float32(scale_factor)

    #now, call the kernel on all agents (GPU)
    #this has to account for
    # - rescaling of positions due to domain size change (not done on CPU, we pick this up after force resolution)
    # - formation of new tethers (already done on CPU)
    # - updating nascent-inserting ideal distances (already done on CPU)
    # - updating effective radii of nascent agents (already done on CPU)
    num_newly_tethered = length(newly_tethered_agent_ixs)
    non_force_position_kernel(
        all_data_metal.positions,
        all_data_metal.next_positions,
        all_data_metal.effective_radii,
        all_data_metal.identifiers,
        all_data_metal.tether_points,
        all_data_metal.substrate_inserting_ideal_dists,
        all_data_metal.newly_tethered_agent_ixs,
        grid_size_metal.dims,
        effective_rad_incs_metal,
        ideal_dist_incs_metal,
        scale_factor_metal,
        num_newly_tethered;
        ndrange=grid_size_metal.num_agents
    )

    #change dims etc HERE (not before kernel call)
    grid_size.dims *= scale_factor
    grid_size_metal.dims *= scale_factor_metal

end




"""
    function _non_force_position_kernel!(
        positions::MtlDeviceVector{Float32},
        next_positions::MtlDeviceVector{Float32},
        effective_radii::MtlDeviceVector{Float32},
        identifiers::MtlDeviceVector{Int},
        tether_points::MtlDeviceVector{Float32},
        substrate_inserting_ideal_dists::MtlDeviceVector{Float32},
        newly_tethered_agent_ixs::MtlDeviceVector{Int},
        dims::SVector{2, Float32},
        effective_rad_incs::SVector{7, Float32},
        ideal_dist_incs::SVector{7, Float32},
        scale_factor::Float32,
        num_newly_tethered::Int
    )

Metal GPU kernel for computing non-force position changes.
"""
@kernel function _non_force_position_kernel!(
    positions::MtlDeviceVector{Float32},
    next_positions::MtlDeviceVector{Float32},
    effective_radii::MtlDeviceVector{Float32},
    identifiers::MtlDeviceVector{Int},
    tether_points::MtlDeviceVector{Float32},
    substrate_inserting_ideal_dists::MtlDeviceVector{Float32},
    newly_tethered_agent_ixs::MtlDeviceVector{Int},
    dims::SVector{2, Float32},
    effective_rad_incs::SVector{7, Float32},
    ideal_dist_incs::SVector{7, Float32},
    scale_factor::Float32,
    num_newly_tethered::Int
    )
    
    # Get sorted index
    sorted_ix = @index(Global)

    # parse this agent's identifier
    agent_data = parse_identifier_metal(identifiers[sorted_ix])
    is_tethered = agent_data[1]
    agent_type_num = agent_data[2]
    is_nascent = agent_data[3]
    nascent_ix = agent_data[5]

    # if this agent just became tethered, set its tether position
    if num_newly_tethered>0
        for i in 1:num_newly_tethered
            if newly_tethered_agent_ixs[i]==sorted_ix
                # remember this agent is now tethered
                is_tethered = 1
                # set tether position to current position
                tether_points[2*sorted_ix-1] = positions[2*sorted_ix-1]
                tether_points[2*sorted_ix] = positions[2*sorted_ix]
                break
            end
        end
    end

    # if this agent is nascent, update its effective radius and ideal distance
    if is_nascent==1
        effective_radii[sorted_ix] += effective_rad_incs[agent_type_num]
        substrate_inserting_ideal_dists[nascent_ix] += ideal_dist_incs[agent_type_num]
    end

    # compute new position and tether position due to domain rescaling
    if is_tethered==1
        agent_to_tether_vec = shortest_vec_metal(
            SVector{2, Float32}(positions[2*sorted_ix-1], positions[2*sorted_ix]),
            SVector{2, Float32}(tether_points[2*sorted_ix-1], tether_points[2*sorted_ix]),
            dims
        )
    end
    next_positions[2*sorted_ix-1] = scale_factor * positions[2*sorted_ix-1]
    next_positions[2*sorted_ix]   = scale_factor * positions[2*sorted_ix]
    if is_tethered==1
        new_tether_pos = SVector{2, Float32}(next_positions[2*sorted_ix-1], next_positions[2*sorted_ix]) + agent_to_tether_vec
        tether_points[2*sorted_ix-1] = new_tether_pos[1]
        tether_points[2*sorted_ix] = new_tether_pos[2]
    end

end