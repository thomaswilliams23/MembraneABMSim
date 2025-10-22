
#define cell and grid types for spatial partitioning

mutable struct SimCell
    cell_z_ix::Int
    agent_ixs::Vector{Int}
end

mutable struct SimGrid
    dims::SVector{2, Float64}
    num_cells::SVector{2, Int}
    num_agents::Int
    coords_to_z_ix::Vector{Int}
    z_ix_to_coords::Vector{Int}
    agent_cell_z_ixs::Vector{Int}
    num_agents_in_cell::Vector{Int}
    start_agents_in_cell::Vector{Int}
    cells::Vector{SimCell}
    has_changed::Bool
end