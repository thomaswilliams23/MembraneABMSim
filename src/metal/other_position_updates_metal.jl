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
function compute_non_force_position_changes_metal!(
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
@kernel function _non_force_position_kernel_metal!(
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