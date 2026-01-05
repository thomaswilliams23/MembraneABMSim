

"""
    update_BAM_subsystem!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat::AllAgentsFlat, params::AllParams, t::Float64)

Updates the BAM subsystem of the system. Specifically, iterates over all (assembled) BamA agents and checks
for changes in insertion state. If necessary, inserts new nascent objects, or changes fully-inserted nascent 
objects to full OMP agents. Also updates polypeptide arrivals.
"""
function update_BAM_subsystem!(agents::AllAgents, grid_size::GridSize, grid::SimGrid, system_flat::AllAgentsFlat, params::AllParams, t::Float64)

    #keep track of nascent OMPs to delete after iteration
    nascent_OMP_ixs_to_delete = Int[]

    #iterate over BamAs in random order
    for BamA in sample(agents.OMP.BamA, length(agents.OMP.BamA), replace=false)

        #ignore any unassembled BamAs
        if BamA.is_part_of_assembled_complex

            #BamA in "free" state
            if BamA.insertion_state == "free"

                #check if there are any polypeptides left
                if agents.num_PP == 0
                    continue
                end

                pr_free_to_bound = 1 - exp(-params.insertion.PP_bind_rate * (agents.num_PP/(prod(grid_size.dims)))*params.system.dt)
                if rand()<pr_free_to_bound
                    BamA.insertion_state = "bound"
                    agents.num_PP -= 1
                end

            #BamA in "bound state"
            elseif BamA.insertion_state == "bound"

                #check if we can attempt insertion this timestep
                attempt_dt = params.insertion.attempt_dt
                attempt_dt_err = 0.1*attempt_dt
                if abs(t/attempt_dt - round(t/attempt_dt))<attempt_dt_err

                    #decide if insertion accepted
                    if params.insertion.type == "guaranteed"
                        insertion_accepted = true
                    elseif params.insertion.type == "lipid-dependent"
                        insertion_accepted = false

                        #find this BamA's sorted index
                        sorted_ix = system_flat.agent_ix_to_sorted_ix[BamA.index]

                        #find grid coords
                        cell_z_ix = grid.agent_cell_z_ixs[sorted_ix]
                        cell_x = grid.z_ix_to_coords[2*cell_z_ix-1]
                        cell_y = grid.z_ix_to_coords[2*cell_z_ix]

                        #get a buffer based on agent radii and sensing radius
                        agent_buffer_radius = params.BamA.radius + params.force.sensing_radius + params.LPS.radius
                        
                        #add error to account for agent movement since last grid sync
                        this_agent_agg_dist = system_flat.agg_dist_since_grid_sync[sorted_ix]
                        agent_buffer_radius += maximum(system_flat.agg_dist_since_grid_sync) + this_agent_agg_dist
                        
                        #convert to grid cell counts
                        cell_buffer_num_x, cell_buffer_num_y = ceil.(Int, agent_buffer_radius./(grid_size.dims./grid_size.num_cells))
                        cell_buffer_num_x = min(cell_buffer_num_x, ceil(Int, grid_size.num_cells[1] / 2))
                        cell_buffer_num_y = min(cell_buffer_num_y, ceil(Int, grid_size.num_cells[2] / 2))
                        
                        #loop over the neighbourhood to find LPS
                        for x_shift=-cell_buffer_num_x:cell_buffer_num_x, y_shift=-cell_buffer_num_y:cell_buffer_num_y
                            neigh_cell_x = mod(cell_x + x_shift -1, grid_size.num_cells[1])+1
                            neigh_cell_y = mod(cell_y + y_shift -1, grid_size.num_cells[2])+1
                            neigh_cell_z_ix = grid.coords_to_z_ix[(neigh_cell_y-1)*grid_size.num_cells[1] + neigh_cell_x]

                            #loop over the agents in this neighbouring cell
                            start_agent = grid.start_agents_in_cell[neigh_cell_z_ix]
                            num_agents = grid.num_agents_in_cell[neigh_cell_z_ix]
                            for sorted_n_ix = start_agent:(start_agent + num_agents - 1)
                                #check if LPS (and not nascent)
                                neighbour_identifier = system_flat.identifiers[sorted_n_ix]
                                if (neighbour_identifier & 1) > 0 || neighbour_identifier > 1000 #i.e. is an OMP || is nascent
                                    continue
                                end

                                #check if close enough
                                LPS_pos = SVector{2, Float64}(system_flat.positions[2*sorted_n_ix-1], system_flat.positions[2*sorted_n_ix])
                                if shortest_distance(BamA.position, LPS_pos, grid_size.dims) < params.BamA.radius + params.force.sensing_radius + params.LPS.radius
                                    insertion_accepted = true
                                    break
                                end
                            end
                            if insertion_accepted
                                break
                            end
                        end
                    else
                        error("Insertion type $(params.insertion.type) not recognised")
                    end

                    #if so, make a new nascent OMP on the edge of the BamA and update this BamA's state
                    if insertion_accepted
                        #make new nascent OMP and update agents and grid structures
                        generate_nascent_OMP_obj!(agents, grid_size, system_flat, BamA, t, params)

                        #update this BamA too
                        BamA.insertion_state = "embedding"
                    end
                end

            #check if an embedding BamA has fully inserted its substrate
            elseif BamA.insertion_state == "embedding"

                #find the corresponding substrate (works because we only have a few nascent agents at a time)
                #TODO improve?
                for (nascent_OMP_ix, nascent_OMP) in enumerate(agents.nascent.nascent_OMP)

                    if nascent_OMP.inserting_agent_index == BamA.index

                        #find this agent's insertion time
                        if nascent_OMP.OMP_type=="OmpA"
                            insertion_time = params.OmpA.radius/params.insertion.OMP_assembly_rate
                        elseif nascent_OMP.OMP_type=="OmpCF"
                            insertion_time = params.OmpCF.radius/params.insertion.OMP_assembly_rate
                        elseif nascent_OMP.OMP_type=="BamA"
                            insertion_time = params.BamA.radius/params.insertion.OMP_assembly_rate
                        elseif nascent_OMP.OMP_type=="LptD"
                            insertion_time = params.LptD.radius/params.insertion.OMP_assembly_rate
                        else
                            error("OMP type $(nascent_OMP.OMP_type) not recognised.")
                        end

                        #check if ready to promote to full agent
                        time_err = 1e-6
                        time_since_insertion = t - nascent_OMP.arrival_time
                        if time_since_insertion > insertion_time + time_err
                            
                            #make a new agent out of this nascent agent and update its identifier
                            is_tethered_init = false
                            tether_point_init = SVector{2, Float64}(0.0, 0.0)
                            is_assembled_init = false
                            insertion_state_init = "free"
                            if nascent_OMP.OMP_type=="OmpA"
                                new_OmpA = OmpAAgent(
                                    nascent_OMP.index,
                                    nascent_OMP.position,
                                    nascent_OMP.arrival_time,
                                    is_tethered_init,
                                    tether_point_init
                                )
                                push!(agents.OMP.OmpA, new_OmpA)
                                sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_OMP.index]
                                system_flat.identifiers[sorted_ix] = make_identifier(;
                                    is_tethered=false, 
                                    agent_type="OmpA", 
                                    is_inserting=false, 
                                    is_nascent=false,
                                    nascent_ix=0
                                )
                            elseif nascent_OMP.OMP_type=="OmpCF"
                                new_OmpCF = OmpCFAgent(
                                    nascent_OMP.index,
                                    nascent_OMP.position,
                                    nascent_OMP.arrival_time
                                )
                                push!(agents.OMP.OmpCF, new_OmpCF)
                                sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_OMP.index]
                                system_flat.identifiers[sorted_ix] = make_identifier(;
                                    is_tethered=false, 
                                    agent_type="OmpCF", 
                                    is_inserting=false, 
                                    is_nascent=false,
                                    nascent_ix=0
                                )
                            elseif nascent_OMP.OMP_type=="BamA"
                                new_BamA = BamAAgent(
                                    nascent_OMP.index,
                                    nascent_OMP.position,
                                    nascent_OMP.arrival_time,
                                    is_assembled_init,
                                    insertion_state_init,
                                )
                                push!(agents.OMP.BamA, new_BamA)
                                sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_OMP.index]
                                system_flat.identifiers[sorted_ix] = make_identifier(;
                                    is_tethered=false, 
                                    agent_type="BamA", 
                                    is_inserting=false, 
                                    is_nascent=false,
                                    nascent_ix=0
                                )
                            elseif nascent_OMP.OMP_type=="LptD"
                                new_LptD = LptDAgent(
                                    nascent_OMP.index,
                                    nascent_OMP.position,
                                    nascent_OMP.arrival_time,
                                    is_tethered_init,
                                    tether_point_init,
                                    is_assembled_init,
                                    insertion_state_init,
                                )
                                push!(agents.OMP.LptD, new_LptD)
                                sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_OMP.index]
                                system_flat.identifiers[sorted_ix] = make_identifier(;
                                    is_tethered=false, 
                                    agent_type="LptD", 
                                    is_inserting=false, 
                                    is_nascent=false,
                                    nascent_ix=0
                                )
                            else
                                error("OMP type $(nascent_OMP.OMP_type) not recognised.")
                            end

                            #update the BamA
                            BamA.insertion_state = "free"

                            #and its identifier
                            inserting_sorted_ix = system_flat.agent_ix_to_sorted_ix[BamA.index]
                            system_flat.identifiers[inserting_sorted_ix] = make_identifier(;
                                is_tethered=false, 
                                agent_type="BamA", 
                                is_inserting=false, 
                                is_nascent=false,
                                nascent_ix=0
                            )

                            #mark this nascent agent for deletion
                            push!(nascent_OMP_ixs_to_delete, nascent_OMP_ix)
                        end

                        break
                    end
                end
            end
        end
    end

    #now update identifiers of all nascent agents and their inserting agents (since their nascent_ix has changed)
    if length(nascent_OMP_ixs_to_delete)>0

        #delete the nascent agents which have been promoted
        deleteat!(agents.nascent.nascent_OMP, sort(nascent_OMP_ixs_to_delete))

        #update_flat_data_nascent_agents!(agents, system_flat)
        grid_size.nascent_promoted = true
    end

    #now check for polypeptide arrivals
    PP_arrival_dist = Poisson(params.insertion.PP_arrival_rate * params.system.dt)
    agents.num_PP += rand(PP_arrival_dist)

end



"""
    update_flat_data_nascent_agents!(agents::AllAgents, system_flat::AllAgentsFlat)

Updates all flat data structure fields relevant to nascent agents, including identifiers, inserting-substrate 
mappings, and ideal distances.
"""
function update_flat_data_nascent_agents!(agents::AllAgents, system_flat::AllAgentsFlat)

    #update identifiers
    nascent_ix = 1
    for nascent_OMP in agents.nascent.nascent_OMP
        #update identifier
        sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_OMP.index]
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type=nascent_OMP.OMP_type, 
            is_inserting=false, 
            is_nascent=true,
            nascent_ix=nascent_ix
        )

        #update inserting agent identifier too
        inserting_agent_index = nascent_OMP.inserting_agent_index
        inserting_agent_sorted_ix = system_flat.agent_ix_to_sorted_ix[inserting_agent_index]
        system_flat.identifiers[inserting_agent_sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type="BamA", 
            is_inserting=true, 
            is_nascent=false,
            nascent_ix=nascent_ix
        )

        #inserting-substrate mappings
        system_flat.nascent_to_substrate_ixs[nascent_ix] = sorted_ix
        system_flat.nascent_to_inserting_ixs[nascent_ix] = inserting_agent_sorted_ix
        system_flat.substrate_inserting_ideal_dists[nascent_ix] = nascent_OMP.ideal_dist_from_inserting_agent

        nascent_ix += 1
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        #update nascent identifier
        sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_LPS.index]
        system_flat.identifiers[sorted_ix] = make_identifier(;
            is_tethered=false, 
            agent_type="LPS", 
            is_inserting=false, 
            is_nascent=true,
            nascent_ix=nascent_ix
        )

        #update inserting agent identifier too
        inserting_agent_index = nascent_LPS.inserting_agent_index
        inserting_agent_sorted_ix = system_flat.agent_ix_to_sorted_ix[inserting_agent_index]
        system_flat.identifiers[inserting_agent_sorted_ix] = make_identifier(;
            is_tethered=true, 
            agent_type="LptD", 
            is_inserting=true, 
            is_nascent=false,
            nascent_ix=nascent_ix
        )

        #inserting-substrate mappings
        system_flat.nascent_to_substrate_ixs[nascent_ix] = sorted_ix
        system_flat.nascent_to_inserting_ixs[nascent_ix] = inserting_agent_sorted_ix
        system_flat.substrate_inserting_ideal_dists[nascent_ix] = nascent_LPS.ideal_dist_from_inserting_agent

        nascent_ix += 1
    end
end



"""
    generate_nascent_OMP_obj!(agents::AllAgents, grid_size::GridSize, system_flat::AllAgentsFlat, BamA::BamAAgent, arrival_time::Float64, params::AllParams)

Given an inserting BamA agent (and other relevant information), generates a new nascent OMP somewhere on the
interior of the edge of the BamA and writes the new nascent OMP into the agents structure. Also updates total
agent count in the grid object.
"""
function generate_nascent_OMP_obj!(agents::AllAgents, grid_size::GridSize, system_flat::AllAgentsFlat, BamA::BamAAgent, arrival_time::Float64, params::AllParams)
    
    #small perturbation to ensure BamA and its substrate are never in the exact same position (messes up forces)
    insertion_eps = 1e-6

    #get index
    new_nascent_OMP_ix = grid_size.num_agents + 1

    #decide type
    OMP_types = ["OmpA", "OmpCF", "BamA", "LptD"]
    OMP_weights = [params.OmpA.insertion_prob, params.OmpCF.insertion_prob, params.BamA.insertion_prob, params.LptD.insertion_prob]
    nascent_OMP_type = sample(OMP_types, ProbabilityWeights(OMP_weights), 1)[1]

    #get actual radius
    if nascent_OMP_type=="OmpA"
        actual_radius = params.OmpA.radius
    elseif nascent_OMP_type=="OmpCF"
        actual_radius = params.OmpCF.radius
    elseif nascent_OMP_type=="BamA"
        actual_radius = params.BamA.radius
    elseif nascent_OMP_type=="LptD"
        actual_radius = params.LptD.radius
    else
        error("Unknown nascent OMP type $nascent_OMP_type")
    end

    #get effective radius (large OMPs start smaller) and initial distance from BamA
    effective_rad_init = min(params.BamA.radius, actual_radius)
    ideal_dist_init = max(insertion_eps, params.BamA.radius - effective_rad_init)

    #generate random insertion position on the edge of the BAM
    theta = 2*pi*rand()
    position_init = BamA.position + ideal_dist_init * SVector{2, Float64}(cos(theta), sin(theta))
    position_init = mod.(position_init, grid_size.dims)

    #create the nascent OMP
    new_nascent_OMP = NascentOMPAgent(
        new_nascent_OMP_ix,
        position_init,
        arrival_time,
        effective_rad_init,
        ideal_dist_init,
        nascent_OMP_type,
        BamA.index
    )

    #update the 

    #push to AllAgents structure
    push!(agents.nascent.nascent_OMP, new_nascent_OMP)

    #update grid size
    grid_size.num_agents += 1
    grid_size.num_agents_changed = true
end



"""
    update_Lpt_subsystem!(agents::AllAgents, grid_size::GridSize, system_flat::AllAgentsFlat, params::AllParams, t::Float64)

Updates the Lpt subsystem of the system. Specifically, iterates over all (assembled) LptD agents and checks
for changes in insertion state. If necessary, generates new nascent LPS or changes fully-inserted nascent LPS
agents to full LPS agents. 
"""
function update_Lpt_subsystem!(agents::AllAgents, grid_size::GridSize, system_flat::AllAgentsFlat, params::AllParams, t::Float64)

    #keep track of nascent LPSs to delete after iteration
    nascent_LPS_ixs_to_delete = Int[]

    #loop LptDs
    for LptD in agents.OMP.LptD

        #only operate on free, assembled LptDs
        if LptD.is_part_of_assembled_complex
            if LptD.insertion_state=="free"

                #decide if a new LPS is to be inserted
                LPS_arrival_dist = Poisson(params.LPS.arrival_rate * params.system.dt)
                insert_new_LPS = (rand(LPS_arrival_dist)>0.0)
                if insert_new_LPS

                    #make new nascent OMP and update agents and grid structures
                    generate_nascent_LPS_obj!(agents, grid_size, LptD, t, params)

                    #update this LptD too
                    LptD.insertion_state = "embedding"
                end

            elseif LptD.insertion_state=="embedding"

                #find substrate LPS (TODO: improve)
                for nascent_LPS_ix = 1:length(agents.nascent.nascent_LPS)
                    nascent_LPS = agents.nascent.nascent_LPS[nascent_LPS_ix]
                    if nascent_LPS.inserting_agent_index == LptD.index

                        #check if this LPS is now fully inserted
                        time_since_insertion = t - nascent_LPS.arrival_time
                        time_err = 1e-6
                        if time_since_insertion > params.LPS.insertion_time + time_err

                            #make a new LPS agent
                            new_LPS = LPSAgent(
                                nascent_LPS.index,
                                nascent_LPS.position,
                                nascent_LPS.arrival_time
                            )
                            push!(agents.LPS, new_LPS)

                            #update its identifier
                            sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_LPS.index]
                            system_flat.identifiers[sorted_ix] = make_identifier(;
                                is_tethered=false, 
                                agent_type="LPS", 
                                is_inserting=false, 
                                is_nascent=false,
                                nascent_ix=0
                            )

                            #update the LptD as well
                            LptD.insertion_state = "free"

                            #and its identifier
                            inserting_sorted_ix = system_flat.agent_ix_to_sorted_ix[LptD.index]
                            system_flat.identifiers[inserting_sorted_ix] = make_identifier(;
                                is_tethered=true, 
                                agent_type="LptD", 
                                is_inserting=false, 
                                is_nascent=false,
                                nascent_ix=0
                            )

                            #mark this nascent agent for deletion
                            push!(nascent_LPS_ixs_to_delete, nascent_LPS_ix)
                        end

                        break
                    end
                end

            end
        end
    end

    #now update identifiers of all nascent agents and their inserting agents (since their nascent_ix has changed)
    if length(nascent_LPS_ixs_to_delete)>0

        #delete the nascent agents which have been promoted
        deleteat!(agents.nascent.nascent_LPS, sort(nascent_LPS_ixs_to_delete))

        #update_flat_data_nascent_agents!(agents, system_flat)
        grid_size.nascent_promoted = true
    end

end


"""
    generate_nascent_LPS_obj!(agents::AllAgents, grid_size::GridSize, LptD::LptDAgent, arrival_time::Float64, params::AllParams)

Given an inserting LptD agent (and other relevant information), generates a new nascent LPS somewhere on the
interior of the edge of the LptD and writes the new nascent LPS into the agents structure. Also updates total
agent count in the grid object.
"""
function generate_nascent_LPS_obj!(agents::AllAgents, grid_size::GridSize, LptD::LptDAgent, arrival_time::Float64, params::AllParams)

    #NOTE: relies on LPS being smaller than LptD (which it is biologically)

    #get index
    new_nascent_LPS_ix = grid_size.num_agents + 1

    #initial ideal distance
    ideal_dist_init = params.LptD.radius - params.LPS.radius

    #generate position
    theta = 2*pi*rand()
    position_init = LptD.position + ideal_dist_init * SVector{2, Float64}(cos(theta), sin(theta))
    position_init = mod.(position_init, grid_size.dims)

    #make new nascent LPS objects
    new_nascent_LPS = NascentLPSAgent(
        new_nascent_LPS_ix,
        position_init,
        arrival_time,
        ideal_dist_init,
        LptD.index
    )

    #push to AllAgents structure
    push!(agents.nascent.nascent_LPS, new_nascent_LPS)

    #update grid
    grid_size.num_agents += 1
    grid_size.num_agents_changed = true
end


"""
    update_tethering_and_assembly!(agents::AllAgents, system_flat_cpu::AllAgentsFlat, params::AllParams)

Iterates through all agents which can be tethered or assembled but haven't been already and checks
for new tether formation or assembly.
"""
function update_tethering_and_assembly!(agents::AllAgents, system_flat_cpu::AllAgentsFlat, params::AllParams)


    #output initialisation
    newly_tethered_agent_ixs = Int[]

    #transition probabilities
    pr_OmpA_tether = 1 - exp(-params.OmpA.tether_rate * params.system.dt)
    pr_BamA_assembly = 1 - exp(-params.BamA.complex_assembly_rate * params.system.dt)
    pr_LptD_assembly = 1 - exp(-params.LptD.complex_assembly_rate * params.system.dt)

    #check OmpAs
    for OmpA in agents.OMP.OmpA
        if !OmpA.is_tethered
            if rand()<pr_OmpA_tether
                OmpA.is_tethered = true
                OmpA.tether_point = OmpA.position
                #remember this (sorted) ix
                sorted_ix = system_flat_cpu.agent_ix_to_sorted_ix[OmpA.index]
                push!(newly_tethered_agent_ixs, sorted_ix)
                #recompute identifier
                sorted_ix = system_flat_cpu.agent_ix_to_sorted_ix[OmpA.index]
                system_flat_cpu.identifiers[sorted_ix] = make_identifier(;
                    is_tethered=true, 
                    agent_type="OmpA", 
                    is_inserting=false, 
                    is_nascent=false,
                    nascent_ix=0
                )
            end
        end
    end

    #check BamAs
    for BamA in agents.OMP.BamA
        if !BamA.is_part_of_assembled_complex
            if rand()<pr_BamA_assembly
                BamA.is_part_of_assembled_complex=true
            end
        end
    end

    #check LptDs
    for LptD in agents.OMP.LptD
        if !LptD.is_part_of_assembled_complex
            if rand()<pr_LptD_assembly
                LptD.is_part_of_assembled_complex=true
                LptD.is_tethered = true
                LptD.tether_point = LptD.position
                #remember this (sorted) ix
                sorted_ix = system_flat_cpu.agent_ix_to_sorted_ix[LptD.index]
                push!(newly_tethered_agent_ixs, sorted_ix)
                #recompute identifier
                sorted_ix = system_flat_cpu.agent_ix_to_sorted_ix[LptD.index]
                system_flat_cpu.identifiers[sorted_ix] = make_identifier(;
                    is_tethered=true, 
                    agent_type="LptD", 
                    is_inserting=false, 
                    is_nascent=false,
                    nascent_ix=0
                )
            end
        end
    end

    return newly_tethered_agent_ixs
end
