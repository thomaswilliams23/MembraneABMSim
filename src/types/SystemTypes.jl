"""
    AllAgentsFlat

Mutable struct storing flat representations of all agents for efficient force calculations. Contains position, radius, identifiers, and mapping vectors.
"""
mutable struct AllAgentsFlat
    positions::Vector{Float64}
    next_positions::Vector{Float64}
    effective_radii::Vector{Float64}
    identifiers::Vector{Int}
    tether_points::Vector{Float64}
    agg_dist_since_grid_sync::Vector{Float64}
    nascent_to_inserting_ixs::Vector{Int}
    nascent_to_substrate_ixs::Vector{Int}
    substrate_inserting_ideal_dists::Vector{Float64}
    agent_ix_to_sorted_ix::Vector{Int}
    sorted_ix_to_agent_ix::Vector{Int}
end

"""
    NascentAddedAreaLookup

Struct holding lookup tables for added area by nascent agents of each type.
"""
struct NascentAddedAreaLookup
    OmpA::Vector{Float64}
    OmpCF::Vector{Float64}
    BamA::Vector{Float64}
    LptD::Vector{Float64}
    LPS::Vector{Float64}
end