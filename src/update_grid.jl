

"""
    rebuild_grid!(grid::SimGrid, agents::AllAgents, sensing_radius::Float64)

Recomputes the spatial grid, potentially changing the number of cells, and reassigning
agents to cells based on their current positions. Also builds two additional fields of
the grid object, `coords_to_z_ix` and `z_ix_to_coords`, which provide a mapping between
cartesian grid coordinates and a dense Morton z-order index.
"""
function rebuild_grid!(grid::SimGrid, agents::AllAgents, sensing_radius::Float64)

    #recompute the number of cells per dimension
    ideal_cell_width = sensing_radius
    num_cells = floor.(Int, grid.dims./ideal_cell_width)
    act_cell_width = grid.dims./num_cells

    #if grid size has changed, rebuild the grid
    if num_cells != grid.num_cells
        grid.num_cells = num_cells
        grid.cells = Vector{SimCell}(undef, prod(num_cells))

        #also need to rebuild the coord to Morton mapping
        coords_to_z_ix, z_ix_to_coords = build_dense_morton(num_cells)
        grid.coords_to_z_ix = coords_to_z_ix
        grid.z_ix_to_coords = z_ix_to_coords
    end

    #create or clear the cells
    start_agent_init = nothing
    num_agents_init = 0
    for cell_z_ix = 1:prod(num_cells)
        agent_ixs_init = Int[]
        grid.cells[cell_z_ix] = SimCell(
            cell_z_ix,
            start_agent_init,
            num_agents_init,
            agent_ixs_init
        )
    end

    #if we have more agents than capacity in the agent_cell_z_ixs vector, resize it
    curr_agent_cell_z_ixs_size = length(grid.agent_cell_z_ixs)
    if grid.num_agents>curr_agent_cell_z_ixs_size
        size_increase_ratio = 1.25
        resize!(grid.agent_cell_z_ixs, ceil(Int, size_increase_ratio*curr_agent_cell_z_ixs_size))
    end

    #now partition agents into cells based on position
    all_agents = Iterators.flatten((agents.OMP.OmpA, agents.OMP.OmpCF, agents.OMP.LptD, agents.OMP.BamA, 
                                    agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
    for agent in all_agents
        agent_ix = agent.index
        cell_x, cell_y = ceil.(Int, agent.position./act_cell_width)
        cell_z_ix = grid.coords_to_z_ix[(cell_y-1)*grid.num_cells[1] + cell_x]

        push!(grid.cells[cell_z_ix].agent_ixs, agent_ix)
        grid.agent_cell_z_ixs[agent_ix] = cell_z_ix
    end
    
    #update numbers of agents in each sim cell and its starting agent
    for cell in grid.cells
        if length(cell.agent_ixs)>0
            cell.start_agent=cell.agent_ixs[1]
            cell.num_agents=length(cell.agent_ixs)
        end
    end

end


"""
    rescale_domain!(agents::AllAgents, grid::SimGrid, params::AllParams, 
                    nascent_added_area_lookup::NascentAddedAreaLookup, t::Float64)

Computes the total amount of area added this time step (as density * total increase in exposed 
nascent object area this time step). Computation of this is trivial with the aid of lookup tables
for nascent added area over the duration of insertion. Once calculated, rescales the grid (membrane) 
dimensions to add that amount of area, and dilates the position of all agents proportionally.
"""
function rescale_domain!(agents::AllAgents, grid::SimGrid, params::AllParams, 
                         nascent_added_area_lookup::NascentAddedAreaLookup, t::Float64)

    #first tally added area this time step
    added_area_this_timestep = 0.0
    for nascent_OMP in agents.nascent.nascent_OMP
        time_since_insertion = t - nascent_OMP.arrival_time
        time_ix = round(Int, time_since_insertion/params.system.dt) + 1
        if nascent_OMP.OMP_type=="OmpA"
            added_area_this_timestep += nascent_added_area_lookup.OmpA[time_ix]
        elseif nascent_OMP.OMP_type=="OmpCF"
            added_area_this_timestep += nascent_added_area_lookup.OmpCF[time_ix]
        elseif nascent_OMP.OMP_type=="BamA"
            added_area_this_timestep += nascent_added_area_lookup.BamA[time_ix]
        elseif nascent_OMP.OMP_type=="LptD"
            added_area_this_timestep += nascent_added_area_lookup.LptD[time_ix]
        else
            error("OMP type $(nascent_OMP.OMP_type) not recognised.")
        end
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        time_since_insertion = t - nascent_LPS.arrival_time
        time_ix = round(Int, time_since_insertion/params.system.dt) + 1
        added_area_this_timestep += nascent_added_area_lookup.LPS[time_ix]
    end


    #return early if no area added
    added_area_err = 1e-10
    if added_area_this_timestep<added_area_err
        return
    end

    #compute scale factor
    prev_area = prod(grid.dims)
    scaled_added_area = added_area_this_timestep/params.system.density
    scale_factor = sqrt((prev_area + scaled_added_area)/prev_area)

    #iterate through each agent and rescale position (if has tether, preserve tether to agent vector)
    not_tethered = false
    for untethered_agent in Iterators.flatten((agents.OMP.OmpCF, agents.OMP.BamA, agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
        rescale_agent!(untethered_agent, grid.dims, scale_factor, not_tethered)
    end
    for tethered_agent in Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD))
        rescale_agent!(tethered_agent, grid.dims, scale_factor, tethered_agent.is_tethered)
    end

    #update dims
    grid.dims *= scale_factor
end


"""
    rescale_agent!(agent::AbstractAgent, old_dims::SVector{2, Float64}, scale_factor::Float64, has_tether::Bool)

Dilates the position of an agent following rescaling by a given `scale_factor`. If the agent has
a tether, the tether point is moved to maintain a fixed vector to the agent's position following
rescaling.
"""
function rescale_agent!(agent::AbstractAgent, old_dims::SVector{2, Float64}, scale_factor::Float64, has_tether::Bool)

    #if this agent has a tether, compute the vector between the agent and its tether
    if has_tether

        if typeof(agent.tether_point)==Vector{Float64}
            println("Offending agent:")
            println(agent)
        end 

        agent_to_tether_vec = shortest_vec(agent.position, agent.tether_point, old_dims)
    end

    #rescale agent position
    agent.position = agent.position * scale_factor

    #if the agent has a tether, move it
    if has_tether
        agent.tether_point = agent.position + agent_to_tether_vec
    end
end