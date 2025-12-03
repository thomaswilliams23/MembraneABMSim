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
        grid_size.grid_size_changed = false
    end

end




"""
    copy_data_to_cpu!(system_flat_cpu::AllAgentsFlat, agents::AllAgents, all_data_metal::AllDataMetal)

Copies position and aggregate distance data from GPU to CPU.
"""
function copy_data_to_cpu_from_metal!(system_flat_cpu::AllAgentsFlat, agents::AllAgents, all_data_metal::AllDataMetal, grid_size_metal::GridSizeMetal)

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
