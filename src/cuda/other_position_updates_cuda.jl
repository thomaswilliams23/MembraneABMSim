"""
    compute_non_force_position_changes!(non_force_position_kernel, agents::AllAgents, all_data_CUDA::AllDataCUDA,
                                       grid_size::GridSize, grid_size_CUDA::GridSizeCUDA,
                                       effective_rad_incs_CUDA::SVector{7, Float32},
                                       ideal_dist_incs_CUDA::SVector{7, Float32},
                                       nascent_added_area_lookup::NascentAddedAreaLookup,
                                       newly_tethered_agent_ixs::Vector{Int},
                                       params::AllParams, t::Float64)

Applies non-force-driven changes to agent positions using a CUDA GPU kernel. Specifically, this handles:
 - rescaling of positions due to domain size change
 - formation of new tethers
 - updating nascent-inserting ideal distances
 - updating effective radii of nascent agents
"""
function compute_non_force_position_changes_CUDA!(
    non_force_position_kernel,
    agents::AllAgents,
    all_data_CUDA::AllDataCUDA,
    grid_size::GridSize,
    grid_size_CUDA::GridSizeCUDA,
    effective_rad_incs_CUDA::SVector{7, Float32},
    ideal_dist_incs_CUDA::SVector{7, Float32},
    nascent_added_area_lookup::NascentAddedAreaLookup,
    newly_tethered_agent_ixs::Vector{Int},
    params::AllParams,
    t::Float64
    )


    #make sure all previous GPU operations are complete
    KernelAbstractions.synchronize(CUDABackend())

    #update positions field on GPU
    copyto!(all_data_CUDA.positions, all_data_CUDA.next_positions[1:2*grid_size_CUDA.num_agents])


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
    scale_factor_CUDA = Float32(scale_factor)

    #now, call the kernel on all agents (GPU)
    #this has to account for
    # - rescaling of positions due to domain size change (not done on CPU, we pick this up after force resolution)
    # - formation of new tethers (already done on CPU)
    # - updating nascent-inserting ideal distances (already done on CPU)
    # - updating effective radii of nascent agents (already done on CPU)
    num_newly_tethered = length(newly_tethered_agent_ixs)
    non_force_position_kernel(
        all_data_CUDA.positions,
        all_data_CUDA.next_positions,
        all_data_CUDA.effective_radii,
        all_data_CUDA.identifiers,
        all_data_CUDA.tether_points,
        all_data_CUDA.substrate_inserting_ideal_dists,
        all_data_CUDA.newly_tethered_agent_ixs,
        grid_size_CUDA.dims,
        effective_rad_incs_CUDA,
        ideal_dist_incs_CUDA,
        scale_factor_CUDA,
        num_newly_tethered;
        ndrange=grid_size_CUDA.num_agents
    )

    #change dims etc HERE (not before kernel call)
    grid_size.dims *= scale_factor
    grid_size_CUDA.dims *= scale_factor_CUDA

end




"""
    function _non_force_position_kernel!(
        positions::CuArray{Float32, 1, CUDA.DeviceMemory},
        next_positions::CuArray{Float32, 1, CUDA.DeviceMemory},
        effective_radii::CuArray{Float32, 1, CUDA.DeviceMemory},
        identifiers::CuArray{Int, 1, CUDA.DeviceMemory},
        tether_points::CuArray{Float32, 1, CUDA.DeviceMemory},
        substrate_inserting_ideal_dists::CuArray{Float32, 1, CUDA.DeviceMemory},
        newly_tethered_agent_ixs::CuArray{Int, 1, CUDA.DeviceMemory},
        dims::SVector{2, Float32},
        effective_rad_incs::SVector{7, Float32},
        ideal_dist_incs::SVector{7, Float32},
        scale_factor::Float32,
        num_newly_tethered::Int
    )

CUDA GPU kernel for computing non-force position changes.
"""
@kernel function _non_force_position_kernel_CUDA!(
    positions::CuArray{Float32, 1, CUDA.DeviceMemory},
    next_positions::CuArray{Float32, 1, CUDA.DeviceMemory},
    effective_radii::CuArray{Float32, 1, CUDA.DeviceMemory},
    identifiers::CuArray{Int, 1, CUDA.DeviceMemory},
    tether_points::CuArray{Float32, 1, CUDA.DeviceMemory},
    substrate_inserting_ideal_dists::CuArray{Float32, 1, CUDA.DeviceMemory},
    newly_tethered_agent_ixs::CuArray{Int, 1, CUDA.DeviceMemory},
    dims::SVector{2, Float32},
    effective_rad_incs::SVector{7, Float32},
    ideal_dist_incs::SVector{7, Float32},
    scale_factor::Float32,
    num_newly_tethered::Int
    )
    
    # Get sorted index
    sorted_ix = @index(Global)

    # parse this agent's identifier
    agent_data = parse_identifier_CUDA(identifiers[sorted_ix])
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
        agent_to_tether_vec = shortest_vec_CUDA(
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