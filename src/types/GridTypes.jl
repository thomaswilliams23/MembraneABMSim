
"""
    GridSize

Mutable struct holding grid dimensions, cell counts, agent counts, and flags for grid state changes. Updates every timestep.
"""
mutable struct GridSize
    dims::SVector{2, Float64}
    num_cells::SVector{2, Int}
    tot_num_cells::Int
    num_agents::Int
    grid_size_changed::Bool
    num_agents_changed::Bool
    nascent_promoted::Bool
end


"""
    SimGrid

Mutable struct containing spatial partitioning data for agents and cells, used for efficient neighbour search and grid management.
"""
mutable struct SimGrid
    coords_to_z_ix::Vector{Int}
    z_ix_to_coords::Vector{Int}
    agent_cell_z_ixs::Vector{Int}
    agent_ixs_in_cell::Vector{Vector{Int}}
    num_agents_in_cell::Vector{Int}
    start_agents_in_cell::Vector{Int}
end