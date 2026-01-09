"""
    get_membrane_size(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Computes the size of the membrane (product of dims).
"""
function get_membrane_size(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    return dims[1] * dims[2]
end


"""
    get_num_agents(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Counts agents of all types in the system.
"""
function get_num_agents(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    return Dict(
        "OmpA" => length(agents.OMP.OmpA),
        "OmpCF" => length(agents.OMP.OmpCF),
        "LptD" => length(agents.OMP.LptD),
        "BamA" => length(agents.OMP.BamA),
        "LPS" => length(agents.LPS)
    )
end


"""
    get_num_OmpA_agents(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Counts the number of OmpA agents in the system.
"""
function get_num_OmpA_agents(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    return length(agents.OMP.OmpA)
end


"""
    get_num_LPS_bordering_OMP(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Counts the number of LPS agents that are bordering any OMP agents.
"""
function get_num_LPS_bordering_OMP(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

    #define bordering distance
    NEIGH_DIST = params.LPS.radius #<- hard-coded, may wish to change

    #count LPS near OMPs
    num_bordering_LPS = 0
    for LPS in agents.LPS
        borders_OMP = false
        for OmpA in agents.OMP.OmpA
            dist = shortest_distance(LPS.position, OmpA.position, dims)
            if dist < params.LPS.radius + params.OmpA.radius + NEIGH_DIST
                num_bordering_LPS += 1
                borders_OMP = true
                break
            end
        end
        if borders_OMP #<- skip to next LPS if already found bordering OMP
            continue
        end
        for OmpCF in agents.OMP.OmpCF
            dist = shortest_distance(LPS.position, OmpCF.position, dims)
            if dist < params.LPS.radius + params.OmpCF.radius + NEIGH_DIST
                num_bordering_LPS += 1
                borders_OMP = true
                break
            end
        end
        if borders_OMP
            continue
        end
        for LptD in agents.OMP.LptD
            dist = shortest_distance(LPS.position, LptD.position, dims)
            if dist < params.LPS.radius + params.LptD.radius + NEIGH_DIST
                num_bordering_LPS += 1
                borders_OMP = true
                break
            end
        end
        if borders_OMP
            continue
        end
        for BamA in agents.OMP.BamA
            dist = shortest_distance(LPS.position, BamA.position, dims)
            if dist < params.LPS.radius + params.BamA.radius + NEIGH_DIST
                num_bordering_LPS += 1
                borders_OMP = true
                break
            end
        end
    end

    return num_bordering_LPS
end


"""
    get_prop_LPS_bordering_OMP(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Computes the proportion of LPS agents that are bordering any OMP agents.
"""
function get_prop_LPS_bordering_OMP(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    num_bordering_LPS = get_num_LPS_bordering_OMP(agents, dims, params)
    total_LPS = length(agents.LPS)
    if total_LPS == 0
        return 0.0
    else
        return num_bordering_LPS / total_LPS
    end
end



"""
    get_LPS_clusters(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Makes a vector of LPS clusters, where each cluster is represented as a vector of agent indices
(that is, indices within the LPS agent vector, not the actual agent indices).
"""
function get_LPS_clusters(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

    #define clustering distance
    CLUSTER_DIST = 2.0 * params.LPS.radius * 1.1 #<- hard-coded, may wish to change

    #keep track of which agents have been assigned to a cluster
    num_LPS = length(agents.LPS)
    assigned_yn = falses(num_LPS)

    clusters = Vector{Vector{Int}}()

    for i in 1:num_LPS
        if assigned_yn[i]
            continue
        end

        #start a new cluster
        new_cluster = [i]
        assigned_yn[i] = true

        #find all LPS connected to this one
        to_check = [i]
        while !isempty(to_check)
            current_ix = pop!(to_check)
            current_agent = agents.LPS[current_ix]

            for j in 1:num_LPS
                if assigned_yn[j]
                    continue
                end
                other_agent = agents.LPS[j]
                dist = shortest_distance(current_agent.position, other_agent.position, dims)
                if dist < CLUSTER_DIST
                    push!(new_cluster, j)
                    assigned_yn[j] = true
                    push!(to_check, j)
                end
            end
        end

        push!(clusters, new_cluster)
    end

    return clusters
end


"""
    get_domain_coverage(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Computes the coverage of the domain by all agents (as a propostion of area). Does so by generating a grid
over the domain and checking which grid points are within the (effective) radius of any agent.
"""
function get_domain_coverage(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    #define grid resolution
    GRID_RES = 0.01 #<- hard-coded, may wish to change

    #number of grid points in each dimension
    num_x_pts = round(Int, dims[1] / GRID_RES)
    num_y_pts = round(Int, dims[2] / GRID_RES)

    #generate grid points
    x_pts = range(0, stop=dims[1], length=num_x_pts)
    y_pts = range(0, stop=dims[2], length=num_y_pts)

    #keep track of covered points
    covered_yn = falses(num_x_pts, num_y_pts)

    #check each grid point against all agents
    for ix in 1:num_x_pts
        for iy in 1:num_y_pts
            grid_point = SVector{2, Float64}(x_pts[ix], y_pts[iy])
            for OmpA in agents.OMP.OmpA
                dist = shortest_distance(grid_point, OmpA.position, dims)
                if dist < params.OmpA.radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
            for OmpCF in agents.OMP.OmpCF
                dist = shortest_distance(grid_point, OmpCF.position, dims)
                if dist < params.OmpCF.radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
            for LptD in agents.OMP.LptD
                dist = shortest_distance(grid_point, LptD.position, dims)
                if dist < params.LptD.radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
            for BamA in agents.OMP.BamA
                dist = shortest_distance(grid_point, BamA.position, dims)
                if dist < params.BamA.radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
            for LPS in agents.LPS
                dist = shortest_distance(grid_point, LPS.position, dims)
                if dist < params.LPS.radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
            for nascent_OMP in agents.nascent.nascent_OMP
                dist = shortest_distance(grid_point, nascent_OMP.position, dims)
                if dist < nascent_OMP.effective_radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
            for nascent_LPS in agents.nascent.nascent_LPS
                dist = shortest_distance(grid_point, nascent_LPS.position, dims)
                if dist < params.LPS.radius
                    covered_yn[ix, iy] = true
                    break
                end
            end
            if covered_yn[ix, iy]
                continue
            end
        end
    end

    #compute coverage proportion
    num_covered_pts = count(covered_yn)
    total_pts = num_x_pts * num_y_pts

    return num_covered_pts / total_pts
end


"""
    get_ideal_domain_coverage(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Computes the ideal coverage of the domain by all agents (as a propostion of area), assuming no overlaps.
"""
function get_ideal_domain_coverage(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    # Calculate total area of all agents
    total_agent_area = 0.0

    # Add area of all OMP agents
    for OmpA in agents.OMP.OmpA
        total_agent_area += π * params.OmpA.radius^2
    end
    for OmpCF in agents.OMP.OmpCF
        total_agent_area += π * params.OmpCF.radius^2
    end
    for LptD in agents.OMP.LptD
        total_agent_area += π * params.LptD.radius^2
    end
    for BamA in agents.OMP.BamA
        total_agent_area += π * params.BamA.radius^2
    end

    # Add area of LPS agents
    for LPS in agents.LPS
        total_agent_area += π * params.LPS.radius^2
    end

    # Add area of nascent agents
    for nascent_OMP in agents.nascent.nascent_OMP
        total_agent_area += π * nascent_OMP.effective_radius^2
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        total_agent_area += π * params.LPS.radius^2
    end

    # Calculate domain area
    domain_area = dims[1] * dims[2]

    return total_agent_area / domain_area
end


"""
    get_cutoff_time(params::AllParams)

Computes the time at which the OMP number plateaus (i.e. no further agents can be inserted).
"""
function get_cutoff_time(params::AllParams)

    # get final OMP number
    out_path = joinpath("out", params.system.output_dir)
    raw_data_dir = joinpath(out_path, "raw_data")
    max_time_ix = round(Int, params.system.t_max / params.system.vis_dt)
    fname_final = joinpath(raw_data_dir, @sprintf("sys_data_%d.jld2", max_time_ix))
    @load fname_final agents dims
    final_OMP_number = get_num_OMP_agents(agents, dims, params)

    # iterate backwards through time points to find when OMP number plateaus
    cutoff_time = params.system.t_max
    for time_ix in (max_time_ix-1):-1:0
        fname_this_time_ix = joinpath(raw_data_dir, @sprintf("sys_data_%d.jld2", time_ix))
        @load fname_this_time_ix agents dims
        OMP_number_this_time = get_num_OMP_agents(agents, dims, params)
        if OMP_number_this_time < final_OMP_number
            cutoff_time = time_ix * params.system.vis_dt
            break
        end
    end

    return cutoff_time
end



"""
    get_squared_displacement(params::AllParams)

Compute the squared displacement of all agents, structured by agent type. Ignores nascent agents. 
Not robust to agent removals.
"""
function get_squared_displacement(params::AllParams)
    
    # get final number of agents
    out_path = joinpath("out", params.system.output_dir)
    raw_data_dir = joinpath(out_path, "raw_data")
    max_time_ix = round(Int, params.system.t_max / params.system.vis_dt)
    fname_final = joinpath(raw_data_dir, @sprintf("sys_data_%d.jld2", max_time_ix))
    @load fname_final agents dims
    final_num_agents = Dict(
        "OmpA" => length(agents.OMP.OmpA),
        "OmpCF" => length(agents.OMP.OmpCF),
        "LptD" => length(agents.OMP.LptD),
        "BamA" => length(agents.OMP.BamA),
        "LPS" => length(agents.LPS)
    )
    num_time_points = max_time_ix + 1

    #initialise
    proportional_origin_positions = Dict(
        "OmpA" => [SVector{2, Float64}(0.0, 0.0) for _ in 1:final_num_agents["OmpA"]],
        "OmpCF" => [SVector{2, Float64}(0.0, 0.0) for _ in 1:final_num_agents["OmpCF"]],
        "LptD" => [SVector{2, Float64}(0.0, 0.0) for _ in 1:final_num_agents["LptD"]],
        "BamA" => [SVector{2, Float64}(0.0, 0.0) for _ in 1:final_num_agents["BamA"]],
        "LPS" => [SVector{2, Float64}(0.0, 0.0) for _ in 1:final_num_agents["LPS"]]
    )
    num_agents_recorded_so_far = Dict(
        "OmpA" => 0,
        "OmpCF" => 0,
        "LptD" => 0,
        "BamA" => 0,
        "LPS" => 0
    )
    squared_displacement = Dict(
        "OmpA" => zeros(Float64, final_num_agents["OmpA"], num_time_points),
        "OmpCF" => zeros(Float64, final_num_agents["OmpCF"], num_time_points),
        "LptD" => zeros(Float64, final_num_agents["LptD"], num_time_points),
        "BamA" => zeros(Float64, final_num_agents["BamA"], num_time_points),
        "LPS" => zeros(Float64, final_num_agents["LPS"], num_time_points)
    )

    #loop time indices
    for time_ix in 0:max_time_ix

        #load in system state at this time point
        fname_this_time_ix = joinpath(raw_data_dir, @sprintf("sys_data_%d.jld2", time_ix))
        @load fname_this_time_ix agents dims

        #loop agent types
        for (agent_type, agent_vector) in zip(
            ["OmpA", "OmpCF", "LptD", "BamA", "LPS"],
            [agents.OMP.OmpA, agents.OMP.OmpCF, agents.OMP.LptD, agents.OMP.BamA, agents.LPS]
        )
            for (agent_ix, agent) in enumerate(agent_vector)
                #record proportional origin position if first time seeing this agent
                if agent_ix > num_agents_recorded_so_far[agent_type]
                    proportional_origin_positions[agent_type][agent_ix] = agent.position ./ dims
                    num_agents_recorded_so_far[agent_type] += 1
                end
                #compute squared displacement
                origin_position = proportional_origin_positions[agent_type][agent_ix] .* dims
                dist = shortest_distance(agent.position, origin_position, dims)
                squared_displacement[agent_type][agent_ix, time_ix+1] = dist^2
            end
        end

    end

    return squared_displacement
end


