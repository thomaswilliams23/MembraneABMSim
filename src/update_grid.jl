

"""
    rebuild_grid!(grid_size::GridSize, grid::SimGrid, agents::AllAgents, sensing_radius::Float64)

Recomputes the spatial grid, potentially changing the number of cells, and reassigning
agents to cells based on their current positions. Also builds two additional fields of
the grid object, `coords_to_z_ix` and `z_ix_to_coords`, which provide a mapping between
cartesian grid coordinates and a dense Morton z-order index.
"""
function rebuild_grid!(grid_size::GridSize, grid::SimGrid, agents::AllAgents, sensing_radius::Float64)

    #recompute the number of cells per dimension
    ideal_cell_width = sensing_radius
    num_cells = floor.(Int, grid_size.dims./ideal_cell_width)
    act_cell_width = grid_size.dims./num_cells

    #if grid size has changed, rebuild the grid
    grid_size.grid_size_changed = false
    if num_cells != grid_size.num_cells

        #flag that grid has changed
        grid_size.grid_size_changed = true

        #rebuild grid fields
        grid_size.num_cells = num_cells
        grid_size.tot_num_cells = prod(num_cells)

        #also need to rebuild the coord to Morton mapping
        coords_to_z_ix, z_ix_to_coords = build_dense_morton(num_cells)
        grid.coords_to_z_ix = coords_to_z_ix
        grid.z_ix_to_coords = z_ix_to_coords

        #check if grid vectors need resizing
        curr_grid_vec_size = length(grid.num_agents_in_cell)
        if grid_size.tot_num_cells > curr_grid_vec_size
            size_increase_ratio = 1.25
            new_vec_size = ceil(Int, size_increase_ratio*grid_size.tot_num_cells)
            resize!(grid.num_agents_in_cell, new_vec_size)
            resize!(grid.start_agents_in_cell, new_vec_size)
            resize!(grid.agent_ixs_in_cell, new_vec_size)
        end

    end

    #clear the cells
    for z_ix = 1:grid_size.tot_num_cells
        grid.agent_ixs_in_cell[z_ix] = Int[]
    end

    #if we have more agents than capacity in the agent_cell_z_ixs vector, resize it
    if grid_size.num_agents>length(grid.agent_cell_z_ixs)
        size_increase_ratio = 1.25
        new_vec_size = ceil(Int, size_increase_ratio*grid_size.num_agents)
        resize!(grid.agent_cell_z_ixs, new_vec_size)
    end

    #now partition agents into cells based on position
    all_agents = Iterators.flatten((agents.OMP.OmpA, agents.OMP.OmpCF, agents.OMP.LptD, agents.OMP.BamA, 
                                    agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
    for agent in all_agents
        agent_ix = agent.index
        
        cell_x, cell_y = ceil.(Int, agent.position ./ act_cell_width)
        cell_x = clamp(cell_x, 1, num_cells[1])
        cell_y = clamp(cell_y, 1, num_cells[2])

        cell_z_ix = grid.coords_to_z_ix[(cell_y-1)*grid_size.num_cells[1] + cell_x]

        push!(grid.agent_ixs_in_cell[cell_z_ix], agent_ix)
        grid.agent_cell_z_ixs[agent_ix] = cell_z_ix
    end
    
    #update numbers of agents in each sim cell and its starting agent
    for z_ix = 1:grid_size.tot_num_cells
        if length(grid.agent_ixs_in_cell[z_ix])>0
            grid.start_agents_in_cell[z_ix] = grid.agent_ixs_in_cell[z_ix][1]
            grid.num_agents_in_cell[z_ix]=length(grid.agent_ixs_in_cell[z_ix])
        else
            grid.start_agents_in_cell[z_ix] = 0
            grid.num_agents_in_cell[z_ix] = 0
        end
    end

end


"""
    rescale_domain!(agents::AllAgents, system_flat::AllAgentsFlat, grid_size::GridSize, params::AllParams, 
                    nascent_added_area_lookup::NascentAddedAreaLookup, t::Float64)

Computes the total amount of area added this time step (as density * total increase in exposed 
nascent object area this time step). Computation of this is trivial with the aid of lookup tables
for nascent added area over the duration of insertion. Once calculated, rescales the grid (membrane) 
dimensions to add that amount of area, and dilates the position of all agents proportionally.
"""
function rescale_domain!(agents::AllAgents, system_flat::AllAgentsFlat, grid_size::GridSize, params::AllParams, 
                         nascent_added_area_lookup::NascentAddedAreaLookup, t::Float64)

    #compute total added area this timestep
    added_area_this_timestep = compute_added_area(agents, params, nascent_added_area_lookup, t)

    #return early if no area added
    added_area_err = 1e-20
    if added_area_this_timestep<added_area_err
        return
    end

    #compute scale factor
    prev_area = prod(grid_size.dims)
    scaled_added_area = added_area_this_timestep/params.system.density
    scale_factor = sqrt((prev_area + scaled_added_area)/prev_area)

    #iterate through each agent and rescale position (if has tether, preserve tether to agent vector)
    for untethered_agent in Iterators.flatten((agents.OMP.OmpCF, agents.OMP.BamA, agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS))
        #update position as agent property
        untethered_agent.position = untethered_agent.position * scale_factor
        #update the position in the flat data structure
        sorted_ix = system_flat.agent_ix_to_sorted_ix[untethered_agent.index]
        system_flat.positions[2*sorted_ix-1] = untethered_agent.position[1]
        system_flat.positions[2*sorted_ix] = untethered_agent.position[2]
    end
    for tethered_agent in Iterators.flatten((agents.OMP.OmpA, agents.OMP.LptD))
        #compute agent-to-tether vector if tethered
        if tethered_agent.is_tethered
            agent_to_tether_vec = shortest_vec(tethered_agent.position, tethered_agent.tether_point, grid_size.dims)
        end
        #update position as agent property
        tethered_agent.position = tethered_agent.position * scale_factor
        #update the position in the flat data structure
        sorted_ix = system_flat.agent_ix_to_sorted_ix[tethered_agent.index]
        system_flat.positions[2*sorted_ix-1] = tethered_agent.position[1]
        system_flat.positions[2*sorted_ix] = tethered_agent.position[2]
        #if tethered, update tether point to preserve agent-to-tether vector
        if tethered_agent.is_tethered
            #update tether point as agent property
            tethered_agent.tether_point = tethered_agent.position + agent_to_tether_vec
            #also update the tether point in the flat data structure
            sorted_ix = system_flat.agent_ix_to_sorted_ix[tethered_agent.index]
            system_flat.tether_points[2*sorted_ix-1] = tethered_agent.tether_point[1]
            system_flat.tether_points[2*sorted_ix] = tethered_agent.tether_point[2]
        end
    end

    #update dims
    grid_size.dims *= scale_factor
end



"""
    compute_added_area(agents::AllAgents, params::AllParams, 
                            nascent_added_area_lookup::NascentAddedAreaLookup, t::Float64)

Helper function to compute the total area added this timestep from all nascent agents.
"""
function compute_added_area(agents::AllAgents, params::AllParams, 
                            nascent_added_area_lookup::NascentAddedAreaLookup, t::Float64)

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
    return added_area_this_timestep
end



"""
    put_grid_in_sorted_order!(grid_size::GridSize, grid::SimGrid, system_flat::AllAgentsFlat)

Uses the ordering in system_flat to redefine grid fields in terms of sorted agent indices.
"""
function put_grid_in_sorted_order!(grid_size::GridSize, grid::SimGrid, system_flat::AllAgentsFlat)

    #redefine agent_cell_z_ixs in terms of sorted indices
    agent_cell_z_ixs_sorted = Vector{Int}(undef, grid_size.num_agents)
    for agent_ix = 1:grid_size.num_agents
        sorted_ix = system_flat.agent_ix_to_sorted_ix[agent_ix]
        agent_cell_z_ixs_sorted[sorted_ix] = grid.agent_cell_z_ixs[agent_ix]
    end
    grid.agent_cell_z_ixs = agent_cell_z_ixs_sorted

    #redefine start_agents_in_cell in terms of sorted indices
    for cell_z_ix = 1:grid_size.tot_num_cells
        if grid.num_agents_in_cell[cell_z_ix]>0
            start_ix = grid.start_agents_in_cell[cell_z_ix]
            sorted_start_ix = system_flat.agent_ix_to_sorted_ix[start_ix]
            grid.start_agents_in_cell[cell_z_ix] = sorted_start_ix
        end

        #actually don't need to reorder agent_ixs in each cell, as we don't use them after building successors list
        # for foo = 1:length(grid.agent_ixs_in_cell[cell_z_ix].agent_ixs)
        #     agent_ix = grid.agent_ixs_in_cell[cell_z_ix][foo]
        #     sorted_ix = system_flat.agent_ix_to_sorted_ix[agent_ix]
        #     grid.agent_ixs_in_cell[cell_z_ix][foo] = sorted_ix
        # end
    end

end


"""
    membrane_contains_hole(agents::AllAgents, grid_size::GridSize, params::AllParams)

Determines whether the membrane contains a hole which can fit a circle of radius at least as large as some threshold.
"""
function membrane_contains_hole(agents::AllAgents, grid_size::GridSize, params::AllParams)

    #if the parameters specify no hole radius, return false immediately
    if isnothing(params.system.max_hole_radius)
        return false
    end

    #otherwise, continue
    PIXEL_GRID_WIDTH = params.system.max_hole_radius / 3.0 #temporary - need fine enough resolution to keep approximation error low, but not so fine that it becomes computationally expensive
    
    @inline function _mark_occupied_pixels!(occupied_grid::BitMatrix, agent_centre::SVector{2, Float64}, agent_radius::Float64, pixel_range::SVector{2, Int}, act_pixel_widths::SVector{2, Float64}, dims::SVector{2, Float64})
        #compute bounding box of pixels to check (accommodating periodic boundaries)
        centre_pixel_pos = act_pixel_widths .* (ceil.(Int, agent_centre ./ act_pixel_widths) .- 0.5)
        for pixel_shift_x in -pixel_range[1]:pixel_range[1]
            for pixel_shift_y in -pixel_range[2]:pixel_range[2]
                #compute pixel centre accommodating periodic boundaries
                this_pixel_pos = centre_pixel_pos .+ act_pixel_widths .* SVector{2, Int}(pixel_shift_x, pixel_shift_y)
                this_pixel_pos = mod.(this_pixel_pos, dims)
                #check if pixel centre is within agent radius
                if shortest_distance(agent_centre, this_pixel_pos, dims) < agent_radius
                    this_pixel_ixes = round.(Int, (this_pixel_pos + 0.5 .* act_pixel_widths) ./ act_pixel_widths)
                    occupied_grid[this_pixel_ixes...] = true
                end
            end
        end
    end
    
    #create a grid of booleans for whether each cell is occupied by an agent
    num_pixels = ceil.(Int, grid_size.dims ./ PIXEL_GRID_WIDTH)
    act_pixel_widths = grid_size.dims ./ num_pixels
    occupied_grid = falses(num_pixels...)

    #iterate over agents and mark all pixels occupied by agents
    OmpA_pixel_range = ceil.(Int, params.OmpA.radius ./ act_pixel_widths)
    for OmpA in agents.OMP.OmpA
        _mark_occupied_pixels!(occupied_grid, OmpA.position, params.OmpA.radius, OmpA_pixel_range, act_pixel_widths, grid_size.dims)
    end
    OmpCF_pixel_range = ceil.(Int, params.OmpCF.radius ./ act_pixel_widths)
    for OmpCF in agents.OMP.OmpCF
        _mark_occupied_pixels!(occupied_grid, OmpCF.position, params.OmpCF.radius, OmpCF_pixel_range, act_pixel_widths, grid_size.dims)
    end
    LptD_pixel_range = ceil.(Int, params.LptD.radius ./ act_pixel_widths)
    for LptD in agents.OMP.LptD
        _mark_occupied_pixels!(occupied_grid, LptD.position, params.LptD.radius, LptD_pixel_range, act_pixel_widths, grid_size.dims)
    end
    BamA_pixel_range = ceil.(Int, params.BamA.radius ./ act_pixel_widths)
    for BamA in agents.OMP.BamA
        _mark_occupied_pixels!(occupied_grid, BamA.position, params.BamA.radius, BamA_pixel_range, act_pixel_widths, grid_size.dims)
    end
    LPS_pixel_range = ceil.(Int, params.LPS.radius ./ act_pixel_widths)
    for LPS in agents.LPS
        _mark_occupied_pixels!(occupied_grid, LPS.position, params.LPS.radius, LPS_pixel_range, act_pixel_widths, grid_size.dims)
    end

    #nascent agents too
    for nascent_OMP in agents.nascent.nascent_OMP
        if nascent_OMP.OMP_type=="OmpA"
            agent_radius = params.OmpA.radius
            pixel_range = ceil.(Int, params.OmpA.radius ./ act_pixel_widths)
        elseif nascent_OMP.OMP_type=="OmpCF"
            agent_radius = params.OmpCF.radius
            pixel_range = ceil.(Int, params.OmpCF.radius ./ act_pixel_widths)
        elseif nascent_OMP.OMP_type=="BamA"
            agent_radius = params.BamA.radius
            pixel_range = ceil.(Int, params.BamA.radius ./ act_pixel_widths)
        elseif nascent_OMP.OMP_type=="LptD"
            agent_radius = params.LptD.radius
            pixel_range = ceil.(Int, params.LptD.radius ./ act_pixel_widths)
        else
            error("OMP type $(nascent_OMP.OMP_type) not recognised.")
        end
        _mark_occupied_pixels!(occupied_grid, nascent_OMP.position, agent_radius, pixel_range, act_pixel_widths, grid_size.dims)
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        _mark_occupied_pixels!(occupied_grid, nascent_LPS.position, params.LPS.radius, LPS_pixel_range, act_pixel_widths, grid_size.dims)
    end
    

    #iterate over all unoccupied pixels and check to see if a hole could be centered there
    inscribed_square_pixel_range = floor.(Int, (params.system.max_hole_radius/sqrt(2)) ./ act_pixel_widths)
    hole_bounding_box_pixel_range = ceil.(Int, params.system.max_hole_radius ./ act_pixel_widths)
    for pixel_x in 1:num_pixels[1]
        for pixel_y in 1:num_pixels[2]
            if !occupied_grid[pixel_x, pixel_y]
                #first check the inscribed SQUARE to prune out easy falses (no distance calculations)
                easy_false = false
                for pixel_shift_x in -inscribed_square_pixel_range[1]:inscribed_square_pixel_range[1]
                    for pixel_shift_y in -inscribed_square_pixel_range[2]:inscribed_square_pixel_range[2]
                        check_pixel_x = mod(pixel_x + pixel_shift_x - 1, num_pixels[1]) + 1
                        check_pixel_y = mod(pixel_y + pixel_shift_y - 1, num_pixels[2]) + 1
                        if occupied_grid[check_pixel_x, check_pixel_y]
                            easy_false = true
                            break
                        end
                    end
                    if easy_false
                        break
                    end
                end
                if !easy_false
                    #then check the full circle of pixels around the center pixel to see if any are occupied
                    this_pixel_pos = act_pixel_widths .* (SVector{2, Int}(pixel_x, pixel_y) .- 0.5)
                    found_hole = true
                    for pixel_shift_x in -hole_bounding_box_pixel_range[1]:hole_bounding_box_pixel_range[1]
                        for pixel_shift_y in -hole_bounding_box_pixel_range[2]:hole_bounding_box_pixel_range[2]
                            #skip the ones in the inscribed square, since we've already checked those
                            if abs(pixel_shift_x)<=inscribed_square_pixel_range[1] && abs(pixel_shift_y)<=inscribed_square_pixel_range[2]
                                continue
                            end
                            check_pixel_x = mod(pixel_x + pixel_shift_x - 1, num_pixels[1]) + 1
                            check_pixel_y = mod(pixel_y + pixel_shift_y - 1, num_pixels[2]) + 1
                            check_pixel_pos = act_pixel_widths .* (SVector{2, Int}(check_pixel_x, check_pixel_y) .- 0.5)
                            if shortest_distance(check_pixel_pos, this_pixel_pos, grid_size.dims) < params.system.max_hole_radius && occupied_grid[check_pixel_x, check_pixel_y]
                                found_hole = false
                                break
                            end
                        end
                        if !found_hole
                            break
                        end
                    end

                    if found_hole
                        return true
                    end
                end
            end
        end
    end

    return false
end