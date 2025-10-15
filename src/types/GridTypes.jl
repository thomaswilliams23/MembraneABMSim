
#define cell and grid types for spatial partitioning

mutable struct SimCell
    cell_z_ix::Int
    start_agent::Union{Int, Nothing}
    num_agents::Int
    agent_ixs::Vector{Int}
end

mutable struct SimGrid
    dims::SVector{2, Float64}
    num_cells::SVector{2, Int}
    num_agents::Int
    coords_to_z_ix::Vector{Int}
    z_ix_to_coords::Vector{Tuple{Int,Int}}
    agent_cell_z_ixs::Vector{Int}
    cells::Vector{SimCell}
end