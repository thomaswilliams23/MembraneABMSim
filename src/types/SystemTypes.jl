
#specifies a type to store flat representations of all agents for efficient force calculation

mutable struct AllAgentsFlat
    positions::Vector{Float64}
    next_positions::Vector{Float64}
    effective_radii::Vector{Float64} 
    actual_radii::Vector{Float64}  
    is_OMP::Vector{Bool}
    is_tethered::Vector{Bool}
    tether_points::Vector{Float64}
    tether_lengths::Vector{Float64}
    substrate_inserting_ixs::Vector{Int}
    substrate_inserting_ideal_dists::Vector{Float64}
    successors::Vector{Int}
    agent_ix_to_sorted_ix::Vector{Int}
    sorted_ix_to_agent_ix::Vector{Int}
end


struct NascentAddedAreaLookup
    OmpA::Vector{Float64}
    OmpCF::Vector{Float64}
    BamA::Vector{Float64}
    LptD::Vector{Float64}
    LPS::Vector{Float64}
end