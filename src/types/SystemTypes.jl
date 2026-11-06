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

"""
    SimState{B<:AbstractSimBackend, R<:AbstractRNG}

Everything that changes or is carried between steps of a simulation: the agents (struct and flat
representations), the spatial grid, the backend, the precomputed nascent lookups, growth and
grid-sync bookkeeping, and the random number generator.
"""
mutable struct SimState{B<:AbstractSimBackend, R<:AbstractRNG}
    agents::AllAgents
    grid_size::GridSize
    grid::SimGrid
    flat::AllAgentsFlat
    backend::B
    nascent_added_area_lookup::NascentAddedAreaLookup
    effective_rad_incs::SVector{7, Float64}
    ideal_dist_incs::SVector{7, Float64}
    allow_membrane_growth::Bool
    steps_since_grid_sync::Int
    max_steps_between_grid_sync::Int
    rng::R
end
