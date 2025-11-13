

"""
    update_nascent_agents!(agents::AllAgents, system_flat::AllAgentsFlat, params::AllParams, t::Float64)

Top level function for updating spring length and effective radius of all nascent objects.
"""
function update_nascent_agents!(agents::AllAgents, system_flat::AllAgentsFlat, params::AllParams, t::Float64)

    #adjust spring distance and size of all nascent objects
    for nascent_OMP in agents.nascent.nascent_OMP
        update_nascent_OMP!(nascent_OMP, params, t)
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        update_nascent_LPS!(nascent_LPS, params, t)
    end

    #update flat data structure
    nascent_ix = 1
    for nascent_OMP in agents.nascent.nascent_OMP
        sorted_ix = system_flat.agent_ix_to_sorted_ix[nascent_OMP.index]
        system_flat.effective_radii[sorted_ix] = nascent_OMP.effective_radius
        system_flat.substrate_inserting_ideal_dists[nascent_ix] = nascent_OMP.ideal_dist_from_inserting_agent
        nascent_ix += 1
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        system_flat.substrate_inserting_ideal_dists[nascent_ix] = nascent_LPS.ideal_dist_from_inserting_agent
        nascent_ix += 1
    end

end


"""
    update_nascent_LPS!(nascent_LPS::NascentLPSAgent, params::AllParams, t::Float64)

Updates the spring length (`ideal_dist_from_inserting_agent`) for a nascent LPS to the correct value
for this time value.
"""
function update_nascent_LPS!(nascent_LPS::NascentLPSAgent, params::AllParams, t::Float64)

    #don't resize nascent LPS (always assumed smaller than LptD), so just extend the spring
    time_since_insertion = max(0.0, t-nascent_LPS.arrival_time)
    insertion_time = params.LPS.insertion_time
    init_dist = params.LptD.radius - params.LPS.radius
    final_dist = params.LptD.radius + params.LPS.radius
    ideal_dist = init_dist + (time_since_insertion/insertion_time)*(final_dist-init_dist)

    #update agent
    nascent_LPS.ideal_dist_from_inserting_agent = ideal_dist
end


"""
    update_nascent_OMP!(nascent_OMP::NascentOMPAgent, params::AllParams, t::Float64)

Updates the spring length (`ideal_dist_from_inserting_agent`) for a nascent OMP to the correct value
for this time value. Also updates the agent's effective radius if it is a large OMP.
"""
function update_nascent_OMP!(nascent_OMP::NascentOMPAgent, params::AllParams, t::Float64)

    #get the actual radius
    if nascent_OMP.OMP_type=="OmpA"
        actual_rad = params.OmpA.radius
    elseif nascent_OMP.OMP_type=="OmpCF"
        actual_rad = params.OmpCF.radius
    elseif nascent_OMP.OMP_type=="BamA"
        actual_rad = params.BamA.radius
    elseif nascent_OMP.OMP_type == "LptD"
        actual_rad = params.LptD.radius
    else
        error("OMP type $(nascent_OMP.OMP_type) not recognised.")
    end

    #calculate insertion time
    time_since_insertion = max(0.0, t-nascent_OMP.arrival_time)
    insertion_time = actual_rad/params.insertion.OMP_assembly_rate

    #if a big agent (bigger than BamA), grow the agent
    if actual_rad>params.BamA.radius
        init_rad = params.BamA.radius
        init_dist = params.BamA.radius - init_rad
        effective_rad = init_rad + (time_since_insertion/insertion_time)*(actual_rad - init_rad)
    else
        init_dist = params.BamA.radius - actual_rad
        effective_rad = nascent_OMP.effective_radius
    end

    #calculate next ideal distance
    final_dist = params.BamA.radius + actual_rad
    ideal_dist = init_dist + (time_since_insertion/insertion_time)*(final_dist - init_dist)

    #update agent
    nascent_OMP.effective_radius = effective_rad
    nascent_OMP.ideal_dist_from_inserting_agent = ideal_dist
end