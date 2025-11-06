
#define cell and grid types for spatial partitioning


# this stuff is small and changes every timestep, so lives on CPU
mutable struct GridSize
    dims::SVector{2, Float64}
    num_cells::SVector{2, Int}
    tot_num_cells::Int
    num_agents::Int
    has_changed::Bool
end

# this stuff is larger and changes less frequently, so can optionally live on GPU
mutable struct SimGrid
    coords_to_z_ix::Vector{Int}
    z_ix_to_coords::Vector{Int}
    agent_cell_z_ixs::Vector{Int}
    agent_ixs_in_cell::Vector{Vector{Int}}
    num_agents_in_cell::Vector{Int}
    start_agents_in_cell::Vector{Int}
end