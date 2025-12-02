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
@kernel function _force_kernel_metal!(
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