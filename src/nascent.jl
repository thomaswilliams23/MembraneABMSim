

"""
    update_nascent_agents!(agents::AllAgents, system_flat::AllAgentsFlat, effective_rad_incs::SVector{7, Float64}, ideal_dist_incs::SVector{7, Float64})

Top level function for updating spring length and effective radius of all nascent objects.
"""
function update_nascent_agents!(agents::AllAgents, system_flat::AllAgentsFlat, effective_rad_incs::SVector{7, Float64}, ideal_dist_incs::SVector{7, Float64})

    #adjust spring distance and size of all nascent objects
    for nascent_OMP in agents.nascent.nascent_OMP
        update_nascent_OMP!(nascent_OMP, effective_rad_incs, ideal_dist_incs)
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        LPS_agent_type_num = 2  #LPS is always type 2 in the increments vectors
        nascent_LPS.ideal_dist_from_inserting_agent += ideal_dist_incs[LPS_agent_type_num]
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
    update_nascent_OMP!(nascent_OMP::NascentOMPAgent, params::AllParams, t::Float64)

Updates the spring length (`ideal_dist_from_inserting_agent`) for a nascent OMP to the correct value
for this time value. Also updates the agent's effective radius if it is a large OMP.
"""
function update_nascent_OMP!(nascent_OMP::NascentOMPAgent, effective_rad_incs::SVector{7, Float64}, ideal_dist_incs::SVector{7, Float64})

    #map OMP type to index in the increments vectors
    agent_type_to_num = Dict(
        "OmpA" => 1,
        "OmpCF" => 3,
        "BamA" => 5,
        "LptD" => 7,
        "LPS" => 2
    )
    agent_type_num = agent_type_to_num[nascent_OMP.OMP_type]
    
    #update agent
    nascent_OMP.effective_radius += effective_rad_incs[agent_type_num]
    nascent_OMP.ideal_dist_from_inserting_agent += ideal_dist_incs[agent_type_num]
end