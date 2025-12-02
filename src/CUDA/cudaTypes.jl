"""
    ParamsCUDA

Immutable struct holding simulation parameters required by the GPU for force calculations.
"""
struct ParamsCUDA
    dt::Float32
    OmpA_radius::Float32
    OmpCF_radius::Float32
    BamA_radius::Float32
    LptD_radius::Float32
    LPS_radius::Float32
    OmpA_tether_radius::Float32
    LptD_tether_radius::Float32
    insertion_mu_attr::Float32
    insertion_mu_rep::Float32
    insertion_k_C::Float32
    sensing_radius::Float32
    mu_attr_LPS_LPS::Float32
    mu_attr_OMP_LPS::Float32
    mu_attr_OMP_OMP::Float32
    mu_rep::Float32
    max_repulsion::Float32
    rho::Float32
    k_C::Float32
    eta::Float32
end



"""
    GridSizeCUDA

Mutable struct holding grid size and agent count information for GPU computations. Lives on CPU and updates every timestep.
"""
mutable struct GridSizeCUDA
    dims::SVector{2, Float32}
    num_cells::SVector{2, Int}
    tot_num_cells::Int
    num_agents::Int
end



"""
    AllDataCUDA

Mutable struct aggregating all vector data required by the GPU for force calculations, including agent and grid data.
Lives on GPU.
"""
mutable struct AllDataCUDA
    #these fields from AllAgentsFlat
    positions::CuArray{Float32, 1, CUDA.DeviceMemory}
    next_positions::CuArray{Float32, 1, CUDA.DeviceMemory}
    effective_radii::CuArray{Float32, 1, CUDA.DeviceMemory}
    identifiers::CuArray{Int, 1, CUDA.DeviceMemory}
    tether_points::CuArray{Float32, 1, CUDA.DeviceMemory}
    agg_dist_since_grid_sync::CuArray{Float32, 1, CUDA.DeviceMemory}
    nascent_to_inserting_ixs::CuArray{Int, 1, CUDA.DeviceMemory}
    nascent_to_substrate_ixs::CuArray{Int, 1, CUDA.DeviceMemory}
    substrate_inserting_ideal_dists::CuArray{Float32, 1, CUDA.DeviceMemory}

    #these fields from SimGrid
    coords_to_z_ix::CuArray{Int, 1, CUDA.DeviceMemory}
    z_ix_to_coords::CuArray{Int, 1, CUDA.DeviceMemory}
    agent_cell_z_ixs::CuArray{Int, 1, CUDA.DeviceMemory}
    num_agents_in_cell::CuArray{Int, 1, CUDA.DeviceMemory}
    start_agents_in_cell::CuArray{Int, 1, CUDA.DeviceMemory}

    #additional field for efficient updating when tethers form
    newly_tethered_agent_ixs::CuArray{Int, 1, CUDA.DeviceMemory}
end


