

"""
    resolve_forces_cpu!(grid::SimGrid, agents::AllAgents, params::AllParams)

Top level function to resolve forces on all agents in the system. Runs with multi-threading by
default, on the GPU as an option.
"""
function resolve_forces_cpu!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat::AllAgentsFlat, params::AllParams)
    
    # write new positions vector (iteration order keeps nearby agents together)
    Threads.@threads for sorted_ix = 1:grid_size.num_agents
        next_position_this_agent = compute_next_position(
            sorted_ix,
            system_flat,
            grid_size,
            grid,
            params
        )
        system_flat.next_positions[2*sorted_ix-1] = next_position_this_agent[1]
        system_flat.next_positions[2*sorted_ix] = next_position_this_agent[2]
    end

    #write new positions into AllAgents structure
    all_agents = Iterators.flatten((agents.OMP.OmpA, agents.OMP.OmpCF, agents.OMP.LptD, agents.OMP.BamA, 
                                    agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
    for agent in all_agents
        agent_ix = agent.index
        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent_ix]
        agent.position = SVector{2, Float64}(system_flat.next_positions[2*sorted_ix-1], system_flat.next_positions[2*sorted_ix])
    end

    #update the flat structure too
    system_flat.positions = copy(system_flat.next_positions)

end


"""
    compile_flat_system_data_cpu!(system_flat::AllAgentsFlat, agents::AllAgents, grid::SimGrid, params::AllParams)

Rebuilds the system_flat (`AllAgentsFlat`) structure used for force calculations. Data is read in
by traversing grid cells in Morton z order for efficiency.
"""
function compile_flat_system_data_cpu!(system_flat::AllAgentsFlat, agents::AllAgents, grid_size::GridSize, grid::SimGrid, params::AllParams)

    #first, if there are any new agents, check the existing vectors are long enough
    curr_vec_capacity = length(system_flat.identifiers)
    if grid_size.num_agents>curr_vec_capacity
        size_increase_ratio = 1.25
        new_vec_size = ceil(Int, size_increase_ratio*grid_size.num_agents)
        resize!(system_flat.positions, 2*new_vec_size)
        resize!(system_flat.next_positions, 2*new_vec_size)
        resize!(system_flat.tether_points, 2*new_vec_size)
        resize!(system_flat.effective_radii, new_vec_size)
        resize!(system_flat.identifiers, new_vec_size)
        resize!(system_flat.agent_ix_to_sorted_ix, new_vec_size)
        resize!(system_flat.sorted_ix_to_agent_ix, new_vec_size)
    end

    #if there is an unusually large number of nascent agents, resize nascent-related vectors
    curr_nascent_capacity = length(system_flat.nascent_to_substrate_ixs)
    num_nascent = length(agents.nascent.nascent_OMP) + length(agents.nascent.nascent_LPS)
    if num_nascent > curr_nascent_capacity
        size_increase_ratio = 1.25
        new_vec_size = ceil(Int, size_increase_ratio * num_nascent)
        resize!(system_flat.nascent_to_substrate_ixs, new_vec_size)
        resize!(system_flat.nascent_to_inserting_ixs, new_vec_size)
        resize!(system_flat.substrate_inserting_ideal_dists, new_vec_size)
    end

    #build the sorted_ixs vectors:

    #traverse grid in Morton z order to populate sorted-unsorted ix maps
    sorted_ix = 1
    for cell_z_ix=1:grid_size.tot_num_cells
        for local_ix = 1:grid.num_agents_in_cell[cell_z_ix]
            this_agent_ix = grid.agent_ixs_in_cell[cell_z_ix][local_ix]
            system_flat.agent_ix_to_sorted_ix[this_agent_ix] = sorted_ix
            system_flat.sorted_ix_to_agent_ix[sorted_ix] = this_agent_ix
            sorted_ix += 1
        end
    end


    #build data vectors:

    #loop through all the agents and populate vectors (janky for SPEED)
    inserting_ix_counter = 1
    nascent_ix_counter = 1
    untethered_val = 0.0
    for agent in agents.OMP.OmpA

        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = agent.tether_point[1]
        system_flat.tether_points[2*sorted_ix] = agent.tether_point[2]
        system_flat.effective_radii[sorted_ix] = params.OmpA.radius
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=agent.is_tethered, 
            agent_type="OmpA", 
            is_inserting=false, 
            is_nascent=false, 
            nascent_ix=0
        )
    end
    for agent in agents.OMP.OmpCF

        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = untethered_val
        system_flat.tether_points[2*sorted_ix] = untethered_val
        system_flat.effective_radii[sorted_ix] = params.OmpCF.radius
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type="OmpCF", 
            is_inserting=false, 
            is_nascent=false, 
            nascent_ix=0
        )
    end
    for agent in agents.OMP.BamA

        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = untethered_val
        system_flat.tether_points[2*sorted_ix] = untethered_val
        system_flat.effective_radii[sorted_ix] = params.BamA.radius
        if agent.insertion_state == "free"
            system_flat.identifiers[sorted_ix] = make_identifier(;
                is_tethered=false, 
                agent_type="BamA", 
                is_inserting=false, 
                is_nascent=false, 
                nascent_ix=0
            )
        else
            system_flat.identifiers[sorted_ix] = make_identifier(;
                is_tethered=false, 
                agent_type="BamA", 
                is_inserting=true, 
                is_nascent=false, 
                nascent_ix=inserting_ix_counter
            )
            inserting_ix_counter += 1
        end
    end
    for agent in agents.OMP.LptD

        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = agent.tether_point[1]
        system_flat.tether_points[2*sorted_ix] = agent.tether_point[2]
        system_flat.effective_radii[sorted_ix] = params.LptD.radius
        if agent.insertion_state == "free"
            system_flat.identifiers[sorted_ix] = make_identifier(;
                is_tethered=agent.is_tethered, 
                agent_type="LptD", 
                is_inserting=false, 
                is_nascent=false, 
                nascent_ix=0
            )
        else
            system_flat.identifiers[sorted_ix] = make_identifier(;
                is_tethered=agent.is_tethered, 
                agent_type="LptD", 
                is_inserting=true, 
                is_nascent=false, 
                nascent_ix=inserting_ix_counter
            )
            inserting_ix_counter += 1
        end
    end
    for agent in agents.LPS

        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = untethered_val
        system_flat.tether_points[2*sorted_ix] = untethered_val
        system_flat.effective_radii[sorted_ix] = params.LPS.radius
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type="LPS", 
            is_inserting=false, 
            is_nascent=false, 
            nascent_ix=0
        )
    end
    for agent in agents.nascent.nascent_OMP

        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = untethered_val
        system_flat.tether_points[2*sorted_ix] = untethered_val
        system_flat.effective_radii[sorted_ix] = agent.effective_radius
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type=agent.OMP_type, 
            is_inserting=false, 
            is_nascent=true, 
            nascent_ix=nascent_ix_counter
        )

        #make identifier for the inserting agent too
        inserting_agent_sorted_index = system_flat.agent_ix_to_sorted_ix[agent.inserting_agent_index]
        system_flat.identifiers[inserting_agent_sorted_index] = make_identifier(;
            is_tethered=false, 
            agent_type="BamA",
            is_inserting=true, 
            is_nascent=false, 
            nascent_ix=nascent_ix_counter
        )

        #add to inserting-substrate mappings
        system_flat.nascent_to_substrate_ixs[nascent_ix_counter] = sorted_ix
        system_flat.nascent_to_inserting_ixs[nascent_ix_counter] = inserting_agent_sorted_index
        system_flat.substrate_inserting_ideal_dists[nascent_ix_counter] = agent.ideal_dist_from_inserting_agent
        nascent_ix_counter += 1
    end
    for agent in agents.nascent.nascent_LPS
    
        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent.index]

        system_flat.positions[2*sorted_ix-1] = agent.position[1]
        system_flat.positions[2*sorted_ix] = agent.position[2]
        system_flat.tether_points[2*sorted_ix-1] = untethered_val
        system_flat.tether_points[2*sorted_ix] = untethered_val
        system_flat.effective_radii[sorted_ix] = params.LPS.radius
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type="LPS", 
            is_inserting=false, 
            is_nascent=true, 
            nascent_ix=nascent_ix_counter
        )

        #make identifier for the inserting agent too
        inserting_agent_sorted_index = system_flat.agent_ix_to_sorted_ix[agent.inserting_agent_index]
        if system_flat.identifiers[inserting_agent_sorted_index] < 0
            inserting_agent_tethered = true
        else
            inserting_agent_tethered = false
        end
        system_flat.identifiers[inserting_agent_sorted_index] = make_identifier(;
            is_tethered=inserting_agent_tethered, 
            agent_type="LPS",
            is_inserting=true, 
            is_nascent=false, 
            nascent_ix=nascent_ix_counter
        )

        #add to inserting-substrate mappings
        system_flat.nascent_to_substrate_ixs[nascent_ix_counter] = sorted_ix
        system_flat.nascent_to_inserting_ixs[nascent_ix_counter] = inserting_agent_sorted_index
        system_flat.substrate_inserting_ideal_dists[nascent_ix_counter] = agent.ideal_dist_from_inserting_agent

        nascent_ix_counter += 1
    end

end


"""
    compute_next_position(sorted_ix::Int, system_flat::AllAgentsFlat, grid::SimGrid, params::AllParams)

Computes the next position of a specified agent after tallying attraction-repulsion forces and 
inserting-substrate forces (if this agent is not inserting something or is being inserted - denoted
by `substrate_inserting_ix` being negative). Returns the next position as a 2-element vector.
"""
function compute_next_position(sorted_ix::Int, system_flat::AllAgentsFlat, grid_size::GridSize, grid::SimGrid, params::AllParams)

    #parse this agent's identifier
    identifier = system_flat.identifiers[sorted_ix]
    is_tethered, agent_type, is_nascent, is_inserting, nascent_ix = parse_identifier(identifier)

    #if inserting or nascent, get the substrate/inserting ix so we can ignore it in attraction/repulsion force calculations
    if is_inserting
        sorted_substrate_inserting_ix = system_flat.nascent_to_substrate_ixs[nascent_ix]
    elseif is_nascent
        sorted_substrate_inserting_ix = system_flat.nascent_to_inserting_ixs[nascent_ix]
    else
        sorted_substrate_inserting_ix = -1
    end

    #tally forces acting on this agent (ignore substrate/inserting agent, if one exists)
    resultant_force = tally_attr_rep_forces(sorted_ix, sorted_substrate_inserting_ix, grid_size, grid, system_flat, params)

    #if this agent has a substrate/inserting agent, compute the spring force between them
    if sorted_substrate_inserting_ix>0
        agent_pos = SVector{2, Float64}(
            system_flat.positions[sorted_ix*2-1], 
            system_flat.positions[sorted_ix*2]
        )
        neigh_pos = SVector{2, Float64}(
            system_flat.positions[sorted_substrate_inserting_ix*2-1], 
            system_flat.positions[sorted_substrate_inserting_ix*2]
        )
        resultant_force += compute_inserting_substrate_force(
            agent_pos,
            neigh_pos,
            grid_size.dims,
            system_flat.substrate_inserting_ideal_dists[nascent_ix],
            params.insertion.mu_attr,
            params.insertion.mu_rep,
            params.insertion.k_C
        )
    end

    #get actual radius
    if is_nascent && agent_type != "LPS"
        if agent_type == "OmpA"
            act_rad = params.OmpA.radius
        elseif agent_type == "OmpCF"
            act_rad = params.OmpCF.radius
        elseif agent_type == "BamA"
            act_rad = params.BamA.radius
        elseif agent_type == "LptD"
            act_rad = params.LptD.radius
        else
            error("Unknown agent type '$agent_type' encountered when getting actual radius for nascent agent.")
        end
    else
        act_rad = system_flat.effective_radii[sorted_ix]
    end

    #compute proposal position from force sum
    agent_pos = SVector{2, Float64}(system_flat.positions[2*sorted_ix-1], system_flat.positions[2*sorted_ix])
    next_pos = agent_pos + (params.system.dt/(params.force.eta*act_rad))*resultant_force
    next_pos = mod.(next_pos, grid_size.dims)

    #check if agent has a tether and if proposal position exceeds tether length
    if is_tethered
        tether_pos = SVector{2, Float64}(system_flat.tether_points[2*sorted_ix-1], system_flat.tether_points[2*sorted_ix])
        if agent_type == "OmpA"
            tether_length = params.OmpA.tether_radius
        elseif agent_type == "LptD"
            tether_length = params.LptD.tether_radius
        else
            error("Tethered agent of type '$agent_type' not supported.")
        end
        if shortest_distance(next_pos, tether_pos, grid_size.dims)>tether_length
            #make next position same as old position
            next_pos = SVector{2, Float64}(system_flat.positions[2*sorted_ix-1], system_flat.positions[2*sorted_ix])
        end
    end

    #return next position
    return next_pos
end


"""
    tally_attr_rep_forces(sorted_ix::Int, substrate_inserting_ix::Int, grid::SimGrid, system_flat::AllAgentsFlat, params::AllParams)

Aggregates all attraction-repulsion forces acting on a specific agent. Returns a 2-element vector containing 
the resultant force on the agent.
"""
function tally_attr_rep_forces(sorted_ix::Int, sorted_substrate_inserting_ix::Int, grid_size::GridSize, grid::SimGrid, system_flat::AllAgentsFlat, params::AllParams)

    #initialise
    resultant_force = SVector{2, Float64}(0.0, 0.0)

    #determine the cell the agent is in
    agent_cell_z_ix = grid.agent_cell_z_ixs[sorted_ix]
    agent_cell_ix = grid.z_ix_to_coords[2*agent_cell_z_ix-1]
    agent_cell_jx = grid.z_ix_to_coords[2*agent_cell_z_ix]

    #calculate how wide around this cell we need to search for neighbours
    max_agent_rad = max(params.OmpA.radius, params.OmpCF.radius, params.BamA.radius, params.LptD.radius, params.LPS.radius)
    agent_buffer_radius = system_flat.effective_radii[sorted_ix] + params.force.sensing_radius + max_agent_rad
    cell_buffer_num_x, cell_buffer_num_y = ceil.(Int, agent_buffer_radius./(grid_size.dims./grid_size.num_cells))

    #cap buffers to avoid double-searching
    cell_buffer_num_x = min(cell_buffer_num_x, ceil(Int, grid_size.num_cells[1] / 2))
    cell_buffer_num_y = min(cell_buffer_num_y, ceil(Int, grid_size.num_cells[2] / 2))

    #loop over Moore neighbourhood
    for x_shift in -cell_buffer_num_x:cell_buffer_num_x, y_shift in -cell_buffer_num_y:cell_buffer_num_y

        #account for periodic boundaries
        neigh_cell_x = mod(agent_cell_ix + x_shift - 1, grid_size.num_cells[1]) + 1
        neigh_cell_y = mod(agent_cell_jx + y_shift - 1, grid_size.num_cells[2]) + 1
        neigh_cell_z_ix = grid.coords_to_z_ix[(neigh_cell_y-1)*grid_size.num_cells[1] + neigh_cell_x]

        #loop over agents
        start_agent = grid.start_agents_in_cell[neigh_cell_z_ix]
        num_agents = grid.num_agents_in_cell[neigh_cell_z_ix]
        for sorted_n_ix = start_agent:(start_agent + num_agents - 1)
            #avoid self-interaction and attr-rep forces between substrate-inserting pairs
            if sorted_n_ix!=sorted_ix && sorted_n_ix!=sorted_substrate_inserting_ix
                
                #work out which type of interaction (for determining mu_attr)
                #(uses bitwise AND on identifiers for speed)
                num_OMPs_in_interaction = 
                    (system_flat.identifiers[sorted_ix] & 1) + 
                    (system_flat.identifiers[sorted_n_ix] & 1)
                if num_OMPs_in_interaction==0
                    mu_attr = params.force.mu_attr_LPS_LPS
                elseif num_OMPs_in_interaction==1
                    mu_attr = params.force.mu_attr_OMP_LPS
                elseif num_OMPs_in_interaction==2
                    mu_attr = params.force.mu_attr_OMP_OMP
                else
                    error("Oops. There seems to be problem deciding which mu_attr to use.")
                end

                #pull out agent position
                agent_position = SVector{2, Float64}(system_flat.positions[2*sorted_ix-1], system_flat.positions[2*sorted_ix])
                neighbour_position = SVector{2, Float64}(system_flat.positions[2*sorted_n_ix-1], system_flat.positions[2*sorted_n_ix])

                #compute force between agent and neighbour
                resultant_force += compute_attr_rep_force(
                    agent_position,
                    system_flat.effective_radii[sorted_ix],
                    neighbour_position,
                    system_flat.effective_radii[sorted_n_ix],
                    grid_size.dims,
                    params.force.sensing_radius,
                    mu_attr,
                    params.force.mu_rep,
                    params.force.max_repulsion,
                    params.force.rho,
                    params.force.k_C
                )

            end
        end
    end

    return resultant_force
end


"""
    compute_attr_rep_force(agent_pos::SVector{2, Float64}, agent_rad::Float64, 
                           neighbour_pos::SVector{2, Float64}, neighbour_rad::Float64, 
                           dims::SVector{2, Int}, sensing_radius::Float64, 
                           mu_attr::Float64, mu_rep::Float64, max_repulsion::Float64,
                           rho::Float64, k_C::Float64) :: SVector{2, Float64}

Computes the attraction-repulsion force between a specific pair of agents as a 2-element vector. Note
that all inputs to this function are simple scalars or vectors (no custom types).
"""
function compute_attr_rep_force(agent_pos::SVector{2, Float64}, agent_rad::Float64, 
                                      neighbour_pos::SVector{2, Float64}, neighbour_rad::Float64, 
                                      dims::SVector{2, Float64}, sensing_radius::Float64, 
                                      mu_attr::Float64, mu_rep::Float64, max_repulsion::Float64,
                                      rho::Float64, k_C::Float64) :: SVector{2, Float64}

    #compute vector and distance between agents
    force_vec = shortest_vec(agent_pos, neighbour_pos, dims)
    dist = norm(force_vec)

    #catch the case where distance is extremely small (avoid div by zero)
    dist_eps = 1e-8
    if dist<dist_eps
        return SVector{2, Float64}(0.0, 0.0)
    end

    #check if within sensing radius
    if dist > agent_rad + neighbour_rad + sensing_radius
        force = SVector{2, Float64}(0.0, 0.0)
    else
        ideal_dist = agent_rad + neighbour_rad

        #repulsion
        force_mag=0
        if dist<=rho*ideal_dist
            force_mag = max_repulsion
        elseif dist<ideal_dist && dist>rho*ideal_dist
            force_mag = mu_rep*ideal_dist*log((dist-rho*ideal_dist)/(1-rho*ideal_dist))
            if force_mag < max_repulsion
                force_mag = max_repulsion
            end
        #attraction
        else
            norm_dist = (dist - ideal_dist)/((1-rho)*ideal_dist)
            force_mag = mu_attr*ideal_dist*norm_dist*exp(-k_C*norm_dist)
        end
        force = (force_mag/dist)*force_vec
    end

    return force
end


"""
    compute_inserting_substrate_force(agent_pos::SVector{2, Float64}, neighbour_pos::SVector{2, Float64},
                                           dims::SVector{2, Float64}, ideal_dist::Float64, 
                                           mu_attr::Float64, mu_rep::Float64, k_C::Float64) :: SVector{2, Float64}
"""
function compute_inserting_substrate_force(agent_pos::SVector{2, Float64}, neighbour_pos::SVector{2, Float64},
                                           dims::SVector{2, Float64}, ideal_dist::Float64, 
                                           mu_attr::Float64, mu_rep::Float64, k_C::Float64) :: SVector{2, Float64}
    
    #handle case where ideal distance is zero or very small
    ideal_dist_eps = 1e-8
    if ideal_dist<ideal_dist_eps
        return SVector{2, Float64}(0.0, 0.0)
    end
    
    #otherwise, compute vector and distance between agents
    force_vec = shortest_vec(agent_pos, neighbour_pos, dims)
    dist = norm(force_vec)

    #handle case where the actual distance is extremely small (avoid div by zero)
    dist_eps = 1e-8
    if dist<dist_eps
        return SVector{2, Float64}(0.0, 0.0)
    end

    #agents too close
    if dist<ideal_dist
        force_mag = mu_rep * ideal_dist * log(dist/ideal_dist)
    #agents too far
    else
        norm_dist = (dist - ideal_dist)/ideal_dist_eps
        force_mag = mu_attr * ideal_dist * norm_dist * exp(-k_C * norm_dist)
    end

    force = (force_mag/dist)*force_vec
    return force

end


"""
    compute_diffusion!(agents::AllAgents, grid::SimGrid, params::AllParams)

Computes the position of all agents following one time step of diffusion. If a proposal move
takes an agent beyond the tether radius of its tether point, the move is rejected.
"""
function compute_diffusion!(agents::AllAgents, grid_size::GridSize, params::AllParams)

    #return early if diffusion temperature is too low
    temp_err = 1e-8
    if params.force.temperature<temp_err
        return
    end

    #precompute all diffusion coefficients and covariance matrices
    diff_coeff_OmpA = params.force.temperature/params.OmpA.radius
    diff_cov_OmpA = 2*diff_coeff_OmpA*params.system.dt*Matrix(I, 2, 2)

    diff_coeff_OmpCF = params.force.temperature/params.OmpCF.radius
    diff_cov_OmpCF = 2*diff_coeff_OmpCF*params.system.dt*Matrix(I, 2, 2)

    diff_coeff_BamA = params.force.temperature/params.BamA.radius
    diff_cov_BamA = 2*diff_coeff_BamA*params.system.dt*Matrix(I, 2, 2)

    diff_coeff_LptD = params.force.temperature/params.LptD.radius
    diff_cov_LptD = 2*diff_coeff_LptD*params.system.dt*Matrix(I, 2, 2)

    diff_coeff_LPS = params.force.temperature/params.LPS.radius
    diff_cov_LPS = 2*diff_coeff_LPS*params.system.dt*Matrix(I, 2, 2)

    #loop each agent type, generate a new position (reject if too far from tether)
    for OmpA in agents.OMP.OmpA
        proposal_distribution = MvNormal(OmpA.position, diff_cov_OmpA)
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        if OmpA.is_tethered
            if shortest_distance(proposal_dest, OmpA.tether_point, grid_size.dims)>params.OmpA.tether_radius
                continue
            end
        end
        OmpA.position = proposal_dest
    end
    for OmpCF in agents.OMP.OmpCF
        proposal_distribution = MvNormal(OmpCF.position, diff_cov_OmpCF)
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        OmpCF.position = proposal_dest
    end
    for BamA in agents.OMP.BamA
        proposal_distribution = MvNormal(BamA.position, diff_cov_BamA)
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        BamA.position = proposal_dest
    end
    for LptD in agents.OMP.LptD
        proposal_distribution = MvNormal(LptD.position, diff_cov_LptD)
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        if LptD.is_tethered
            if shortest_distance(proposal_dest, LptD.tether_point, grid_size.dims)>params.LptD.tether_radius
                continue
            end
        end
        LptD.position = proposal_dest
    end
    for LPS in agents.LPS
        proposal_distribution = MvNormal(LPS.position, diff_cov_LPS)
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        LPS.position = proposal_dest
    end
    for nascent_OMP in agents.nascent.nascent_OMP
        if nascent_OMP.OMP_type == "OmpA"
            proposal_distribution = MvNormal(nascent_OMP.position, diff_cov_OmpA)
        elseif nascent_OMP.OMP_type == "OmpCF"
            proposal_distribution = MvNormal(nascent_OMP.position, diff_cov_OmpCF)
        elseif nascent_OMP.OMP_type == "BamA"
            proposal_distribution = MvNormal(nascent_OMP.position, diff_cov_BamA)
        elseif nascent_OMP.OMP_type == "LptD"
            proposal_distribution = MvNormal(nascent_OMP.position, diff_cov_LptD)
        else
            error("Nascent OMP type $(nascent_OMP.OMP_type) not recognised.")
        end
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        nascent_OMP.position = proposal_dest
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        proposal_distribution = MvNormal(nascent_LPS.position, diff_cov_LPS)
        proposal_dest = rand(proposal_distribution)
        proposal_dest = mod.(proposal_dest, grid_size.dims)
        nascent_LPS.position = proposal_dest
    end
end
