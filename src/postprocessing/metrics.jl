"""
    get_membrane_size(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)

Computes the size of the membrane (product of dims).
"""
function get_membrane_size(agents::AllAgents, dims::SVector{2, Float64}, params::AllParams)
    return dims[1] * dims[2]
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